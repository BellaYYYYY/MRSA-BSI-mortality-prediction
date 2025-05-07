import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from joblib import dump
import shap
import optuna
from optuna.samplers import TPESampler
from probatus.feature_elimination import ShapRFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score, 
                            precision_score, confusion_matrix, roc_curve, log_loss, 
                            brier_score_loss)
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm

output_dir = "output/xgb/XGBoost_outputs_90d/90d_shaprfecv_final"
os.makedirs(output_dir, exist_ok=True)
model_filename = "xgb_shaprfecv_90d.joblib"

SEED = 42


# Features configuration
features = [
    'Sepsis', 'Sex', 'Age', 
    'mv', 'vasopressor', 'rrt', 'ICU_admission', 
    'HA_MRSA', 
    'delta SOFA score', 'prehosp_SOFA_total', 'hospital_SOFA_total', 
    'CCI', 
    'mi', 'chf', 'pvd', 'cevd', 'dementia', 'cpd', 'rheumd', 'pud', 
    'mld', 'diab', 'diabwc', 'hp', 'rend', 'canc', 'msld', 'metacanc', 'aids',
    'Number of Surgeries', 'Number_emergency_surgeries', 
    'df_bili_Std Lab Result', 
    'df_bili_Lab Result of Last Timepoint', 
    'df_bili_Lab Result of First Timepoint',
    'df_crea_Std Lab Result', 
    'df_crea_Lab Result of Last Timepoint', 
    'df_crea_Lab Result of First Timepoint', 
    'df_plate_Std Lab Result', 
    'df_plate_Lab Result of Last Timepoint', 
    'df_plate_Lab Result of First Timepoint', 
    'virus_coinfection',
    'All_CNS', 'All_Eye/Dental/ENT',
    'All_Gastrointestinal/Peritoneal', 'All_Genital',
    'All_Lower respiratory', 'All_Musculoskeletal',
    'All_Other_unknown_sites', 'All_Prosthesis/Lines', 'All_Skin/Wound',
    'All_Systemic', 'All_Urinary',
    'MRSA_CNS', 'MRSA_Eye/Dental/ENT',
    'MRSA_Gastrointestinal/Peritoneal', 'MRSA_Genital',
    'MRSA_Lower respiratory', 'MRSA_Musculoskeletal',
    'MRSA_Other_unknown_sites', 'MRSA_Prosthesis/Lines', 'MRSA_Skin/Wound',
    'MRSA_Urinary'
]

targets = ['Mortality_90d']


def load_data():
    train_year = pd.read_csv('datasets/train_test_90d/train_df_90d.csv')
    test_year = pd.read_csv('datasets/train_test_90d/test_df_90d.csv')
        
    X_train = train_year[features]
    y_train = train_year[targets].values.flatten()
    X_test = test_year[features]
    y_test = test_year[targets].values.flatten()
        
    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    scale_pos_weight = negative_count / positive_count
        
        
    return X_train, y_train, X_test, y_test, scale_pos_weight
    

def perform_feature_selection(X_train, y_train, features, scale_pos_weight):
    with open('output/xgb/XGBoost_outputs_90d/90d_nofs_final/best_trial_params.json', 'r') as f:
        fixed_params_fs = json.load(f)

    estimator = XGBClassifier(**fixed_params_fs)
        
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        
    shaprfecv = ShapRFECV(
        estimator,
        step=1,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=SEED,
        min_features_to_select=10
    )
        
    shap_elimination = shaprfecv.fit(X_train, y_train)
    selected_features = shap_elimination.get_reduced_features_set(num_features='best')
    n_features_selected = len(selected_features)
            
    selected_features_df = pd.DataFrame(selected_features, columns=['Selected Features'])
    output_file_path = os.path.join(output_dir, 'selected_features.csv')
    selected_features_df.to_csv(output_file_path, index=False)
        
    return selected_features, shaprfecv
        

def objective(trial):
    # Parameters to be tuned by Optuna
    tune_params = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }
    
    param = {**fixed_params, **tune_params}

    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Define the model with the parameters
    model = xgb.XGBClassifier(**param)
    scores = cross_val_score(model, X_train_final, y_train_final, cv=skf, scoring='neg_log_loss', n_jobs=1)

    mean_score = scores.mean()
    return mean_score


def gen_performance_metrics(y_true, y_preds, probabilities, model_name, round_to=3):
        accuracy = round(accuracy_score(y_true, y_preds), round_to)
        f1 = round(f1_score(y_true, y_preds), round_to)
        recall = round(recall_score(y_true, y_preds), round_to)
        roc_auc = round(roc_auc_score(y_true, probabilities), round_to)
        precision = round(precision_score(y_true, y_preds), round_to)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
        specificity = round(tn / (tn + fp), round_to) if (tn + fp) > 0 else 0
        brier_score = brier_score_loss(y_true, probabilities)

        metrics = pd.DataFrame([[model_name, accuracy, f1, recall, specificity, roc_auc, precision, round(brier_score, round_to)]], 
                            columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Specificity', 'ROC AUC Score', 'Precision', 'Brier Score'])
        
        return metrics
    

if __name__ == "__main__":        
        X_train, y_train, X_test, y_test, scale_pos_weight = load_data()
        
        selected_features, shaprfecv = perform_feature_selection(X_train, y_train, features, scale_pos_weight)
        
        X_train_final = X_train[selected_features]
        y_train_final = y_train
        X_test_final = X_test[selected_features]
        y_test_final = y_test
                
        fixed_params = {
            "verbosity": 0,
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "booster": "gbtree",
            "n_jobs": -1,
            "objective": "binary:logistic",
        }
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
        study.optimize(objective, n_trials=300, n_jobs=-1, show_progress_bar=True)
                        
        # Get best parameters
        best_trial = study.best_trial
                    
        # Combine parameters for final model
        full_params = {**fixed_params, **best_trial.params}
        
        # Export parameters
        params_json_path = os.path.join(output_dir, 'best_trial_params.json')
        with open(params_json_path, 'w') as f:
            json.dump(full_params, f, indent=2)
                    
        # Create the model with the optimized parameters
        best_model = xgb.XGBClassifier(**full_params)
        
        # Fit the model on the full training dataset
        best_model.fit(X_train_final, y_train_final)
                
        # Save the model
        model_path = os.path.join(output_dir, model_filename)
        dump(best_model, model_path)
        
        # Evaluate on test set
        y_preds = best_model.predict(X_test_final)
        probabilities = best_model.predict_proba(X_test_final)[:, 1]
        
        # Performance metrics
        model_name = 'xgb_90d_fs'
        metrics_df = gen_performance_metrics(y_test_final, y_preds, probabilities, model_name, round_to=3)
        metrics_df.to_csv(os.path.join(output_dir, f'metrics_{model_name}.csv'), index=False)
                
        
