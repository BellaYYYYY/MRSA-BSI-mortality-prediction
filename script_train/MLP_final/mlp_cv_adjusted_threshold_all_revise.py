import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_curve, accuracy_score, f1_score, recall_score, 
    precision_score, confusion_matrix, roc_auc_score
)
from sklearn.base import clone, BaseEstimator, TransformerMixin

class PostProcessingImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lab_prefixes_ = ['df_bili', 'df_crea', 'df_plate']
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        for pref in self.lab_prefixes_:
            last = f"{pref}_Lab Result of Last Timepoint"
            first = f"{pref}_Lab Result of First Timepoint"
            diff = f"{pref}_Last - First"
            if all(col in X.columns for col in [last, first, diff]):
                mask_last = X[last].isna() & X[first].notna() & X[diff].notna()
                X.loc[mask_last, last] = X.loc[mask_last, first] + X.loc[mask_last, diff]
        return X

def calculate_liu_threshold(y_true, probabilities):
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    liu_stat = tpr * (1 - fpr)
    best_idx = np.argmax(liu_stat)
    return thresholds[best_idx]

def get_metrics(y_true, probabilities, threshold=0.5):
    y_preds = (probabilities >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
    
    accuracy = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    recall = recall_score(y_true, y_preds, zero_division=0)
    precision = precision_score(y_true, y_preds, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc = roc_auc_score(y_true, probabilities)
    
    return {
        'Threshold': round(threshold, 4),
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'Precision (PPV)': precision,
        'NPV': npv,
        'ROC AUC': auc
    }

def run_one_fold(fold_idx, train_index, val_index, X_train, y_train, model, model_name):    
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model_cv = clone(model)
    model_cv.fit(X_tr, y_tr)
    
    probs = model_cv.predict_proba(X_val)[:, 1]
    
    fold_metrics = get_metrics(y_val, probs, threshold=0.5)
    fold_metrics['Model'] = model_name
    fold_metrics['Repeat'] = (fold_idx // 10) + 1
    fold_metrics['Fold'] = (fold_idx % 10) + 1
    
    return probs, y_val.tolist(), fold_metrics

def main():
    train_data_path = 'datasets/train_test_202511_80/train.csv'
    test_data_path = 'datasets/train_test_202511_80/test.csv'
    model_dir = 'output/mlp/'
    
    os.makedirs(model_dir, exist_ok=True)

    models_info = [
        {'file': 'output/mlp/mlp_fs_1y_final/mlp_1y_final.joblib', 'target': 'Mortality_1year', 'name': '1y'},
        {'file': 'output/mlp/mlp_fs_28d_final/mlp_28d_final.joblib', 'target': 'Mortality_28day', 'name': '28d'},
        {'file': 'output/mlp/mlp_fs_90d_final/mlp_90d_final.joblib', 'target': 'Mortality_90day', 'name': '90d'}
    ]
    
    try:
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
    except FileNotFoundError as e:
        return

    all_results = []
    all_cv_metrics = []
    all_cv_preds = []
    
    for info in models_info:
        model_path = info['file']
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        else:
            continue
            
        target = info['target']
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]
        
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=68)
        
        cv_results = joblib.Parallel(n_jobs=2)(
            joblib.delayed(run_one_fold)(i, t_idx, v_idx, X_train, y_train, model, info['name'])
            for i, (t_idx, v_idx) in enumerate(rskf.split(X_train, y_train))
        )
        
        all_fold_probs, all_fold_y = [], []
        fold_repeat_list, fold_idx_list = [], []
        
        for probs, y_vals, fold_metrics in cv_results:
            all_fold_probs.extend(probs)
            all_fold_y.extend(y_vals)
            fold_repeat_list.extend([fold_metrics['Repeat']] * len(probs))
            fold_idx_list.extend([fold_metrics['Fold']] * len(probs))
            all_cv_metrics.append(fold_metrics)
            
        model_cv_preds = pd.DataFrame({
            'Model': info['name'],
            'Repeat': fold_repeat_list,
            'Fold': fold_idx_list,
            'y_true': all_fold_y,
            'y_prob': all_fold_probs
        })
        all_cv_preds.append(model_cv_preds)
        
        liu_threshold = calculate_liu_threshold(all_fold_y, all_fold_probs)
        
        test_probs = model.predict_proba(X_test)[:, 1]
        
        for adj_type, thresh in [('Unadjusted', 0.5), ('Liu Adjusted', liu_threshold)]:
            m = get_metrics(y_test, test_probs, threshold=thresh)
            m.update({'Model Path': model_path, 'Target': target, 'Adjustment': adj_type})
            all_results.append(m)
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(model_dir, 'mlp_test_evaluation.csv'), index=False)
        pd.DataFrame(all_cv_metrics).to_csv(os.path.join(model_dir, 'mlp_cv_metrics.csv'), index=False)
        pd.concat(all_cv_preds).to_csv(os.path.join(model_dir, 'mlp_cv_predictions.csv'), index=False)

if __name__ == "__main__":
    main()
