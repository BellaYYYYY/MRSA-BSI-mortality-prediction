import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold
from sklearn.metrics import (
    roc_curve, accuracy_score, f1_score, recall_score, 
    precision_score, confusion_matrix, roc_auc_score
)
from sklearn.base import clone

def calculate_liu_threshold(y_true, probabilities):
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    liu_stat = tpr * (1 - fpr)
    best_idx = np.argmax(liu_stat)
    return thresholds[best_idx]


def get_metrics(y_true, probabilities, threshold=0.5):
    y_preds = (probabilities >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc = roc_auc_score(y_true, probabilities)
    
    return {
        'Threshold': threshold,
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
    fold_metrics['Repeat'] = fold_idx // 10 + 1
    fold_metrics['Fold'] = fold_idx % 10 + 1
    
    return probs, y_val.tolist(), fold_metrics

def main():
    train_data_path = os.path.join('datasets/train_test_202511_80/train.csv')
    test_data_path = os.path.join('datasets/train_test_202511_80/test.csv')
    model_dir = 'output/randomforest_output/'
    models_info = [
        {'file': 'output/randomforest_output/rf_1y/rf_FS_1y_final/FS_1y.joblib', 'target': 'Mortality_1year', 'name': '1y'},
        {'file': 'output/randomforest_output/rf_28d/rf_FS_28d_final_/FS_28d.joblib', 'target': 'Mortality_28day', 'name': '28d'},
        {'file': 'output/randomforest_output/rf_90d/rf_FS_90d_final/FS_90d.joblib', 'target': 'Mortality_90day', 'name': '90d'}
    ]
    

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
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
        
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]
        
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=68)
        cv_results = joblib.Parallel(n_jobs=2)(
            joblib.delayed(run_one_fold)(fold_idx, train_idx, val_idx, X_train, y_train, model, info['name'])
            for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train))
        )
        
        all_fold_probs = []
        all_fold_y = []
        all_fold_repeats = []
        all_fold_folds = []
        
        for probs, y_vals, fold_metrics in cv_results:
            num_samples = len(probs)
            all_fold_probs.extend(probs)
            all_fold_y.extend(y_vals)
            all_fold_repeats.extend([fold_metrics['Repeat']] * num_samples)
            all_fold_folds.extend([fold_metrics['Fold']] * num_samples)
            all_cv_metrics.append(fold_metrics)
            
        model_cv_preds = pd.DataFrame({
            'Model': info['name'],
            'Repeat': all_fold_repeats,
            'Fold': all_fold_folds,
            'y_true': all_fold_y,
            'y_prob': all_fold_probs
        })
        all_cv_preds.append(model_cv_preds)
        
        liu_threshold = calculate_liu_threshold(all_fold_y, all_fold_probs)
        
        test_probs = model.predict_proba(X_test)[:, 1]
        
        unadj_metrics = get_metrics(y_test, test_probs, threshold=0.5)
        unadj_metrics['Model Path'] = info['file']
        unadj_metrics['Target'] = target
        unadj_metrics['Adjustment'] = 'Unadjusted'
        all_results.append(unadj_metrics)
        adj_metrics = get_metrics(y_test, test_probs, threshold=liu_threshold)
        adj_metrics['Model Path'] = info['file']
        adj_metrics['Target'] = target
        adj_metrics['Adjustment'] = 'Liu Adjusted'
        all_results.append(adj_metrics)
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        cols = ['Model Path', 'Target', 'Adjustment', 'Threshold', 'Accuracy', 'F1 Score', 'Recall (Sensitivity)', 'Specificity', 'Precision (PPV)', 'NPV', 'ROC AUC']
        results_df = results_df[cols]
        
        output_path = os.path.join(model_dir, 'rf_calibration_results_liu_vs_unadjusted.csv')
        results_df.to_csv(output_path, index=False)
        
        cv_metrics_df = pd.DataFrame(all_cv_metrics)
        cv_metrics_path = os.path.join(model_dir, 'rf_cv_calibration_metrics.csv')
        cv_metrics_df.to_csv(cv_metrics_path, index=False)
        
        cv_preds_df = pd.concat(all_cv_preds, ignore_index=True)
        cv_preds_path = os.path.join(model_dir, 'rf_cv_predictions_all_models.csv')
        cv_preds_df.to_csv(cv_preds_path, index=False)

if __name__ == "__main__":
    main()
