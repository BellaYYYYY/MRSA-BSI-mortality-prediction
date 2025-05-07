import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from joblib import dump
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score, 
                            precision_score, confusion_matrix, roc_curve, log_loss, 
                            brier_score_loss)
from datetime import datetime

output_dir = "output/xgb/XGBoost_outputs_28d/28d_nofs_final"
os.makedirs(output_dir, exist_ok=True)
model_filename = "xgb_nofs_28d.joblib"

SEED = 123

 
features = [
    'Sepsis', 'Sex', 'Age', 
    'mv', 'vasopressor', 'rrt', 'ICU_admission', 
    'HA_MRSA', 'delta SOFA score', 'prehosp_SOFA_total', 'hospital_SOFA_total', 
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

targets = ['Mortality_28day']

def load_data():
        train_year = pd.read_csv('datasets/train_test_28d/train_df_28d.csv')
        test_year = pd.read_csv('datasets/train_test_28d/test_df_28d.csv')

        X_train = train_year[features]
        y_train = train_year[targets].values.flatten()
        X_test = test_year[features]
        y_test = test_year[targets].values.flatten()
        
        positive_count = np.sum(y_train == 1)
        negative_count = np.sum(y_train == 0)
        scale_pos_weight = negative_count / positive_count
                
        return X_train, y_train, X_test, y_test, scale_pos_weight
    


def objective(trial):
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

        model = xgb.XGBClassifier(**param)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='neg_log_loss')
        
        return scores.mean()
    

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
        
        
        best_trial = study.best_trial
                    
        full_params = {**fixed_params, **best_trial.params}
        
        params_json_path = os.path.join(output_dir, 'best_trial_params.json')
        with open(params_json_path, 'w') as f:
            json.dump(full_params, f, indent=2)
                    
        best_model = xgb.XGBClassifier(**full_params)
        
        best_model.fit(X_train, y_train)
                
        model_path = os.path.join(output_dir, model_filename)
        dump(best_model, model_path)
        
        y_preds = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        
        model_name = 'xgb_28d_nofs'
        metrics_df = gen_performance_metrics(y_test, y_preds, probabilities, model_name, round_to=3)
        metrics_df.to_csv(os.path.join(output_dir, f'metrics_{model_name}.csv'), index=False)
                        
