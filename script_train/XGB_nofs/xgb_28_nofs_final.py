import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import os
import sys
import json
from joblib import dump
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score, 
                            precision_score, confusion_matrix, log_loss)
import logging
from datetime import datetime

output_dir = "output/xgb_work/XGBoost_outputs_28d/28d_nofs_final_noweight"
os.makedirs(output_dir, exist_ok=True)
model_filename = "xgb_nofs_28d_noweight.joblib"
SEED = 25

NUM_GPUS = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "xgb_28d_nofs.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

features = [
    'Sepsis', 'Sex', 
    'mv', 'vasopressor', 'rrt', 'ICU_admission', 
    'HA_MRSA', 
    'mi',
    'chf',
    'pvd',
    'cevd',
    'dementia',
    'cpd',
    'rheumd',
    'pud',
    'mld',
    'diab',
    'diabwc',
    'hp',
    'rend',
    'canc',
    'msld',
    'metacanc',
    'aids',
    'delta SOFA score', 'prehosp_SOFA_total', 'hospital_SOFA_total', 
    'CCI','Age',
    'df_bili_Lab Result of Last Timepoint', 
    'df_bili_Lab Result of First Timepoint',
    'df_crea_Lab Result of Last Timepoint', 
    'df_crea_Lab Result of First Timepoint', 
    'df_plate_Lab Result of Last Timepoint', 
    'df_plate_Lab Result of First Timepoint',
    'df_plate_Last - First',
    'df_crea_Last - First',
    'df_bili_Last - First',
    'MRSA_Eye/Dental/ENT',
    'MRSA_Gastrointestinal/Peritoneal', 'MRSA_Genital',
    'MRSA_Lower respiratory', 'MRSA_Musculoskeletal',
    'MRSA_Other_unknown_sites', 'MRSA_Prosthesis/Lines', 'MRSA_Skin/Wound',
    'MRSA_Urinary', 
    'All_Eye/Dental/ENT', 'All_Gastrointestinal/Peritoneal',
    'All_Genital', 'All_Lower respiratory', 'All_Musculoskeletal',
    'All_Other_unknown_sites', 'All_Prosthesis/Lines', 'All_Skin/Wound',
    'All_Systemic', 'All_Urinary'
]

targets = ['Mortality_28day']

def load_data():
    try:
        logger.info("Loading datasets...")
        train_year = pd.read_csv('datasets/train_test_202511_80/train.csv')
        test_year = pd.read_csv('datasets/train_test_202511_80/test.csv')
              
        X_train = train_year[features]
        y_train = train_year[targets].values.flatten()
        X_test = test_year[features]
        y_test = test_year[targets].values.flatten()
        
        weight_pos_scale = 1
        
        logger.info(f"Data loaded successfully. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"Weight: {weight_pos_scale:.4f}")
        
        return X_train, y_train, X_test, y_test, weight_pos_scale
    
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def objective(trial):
    try:
        gpu_id = trial.number % NUM_GPUS
        
        tune_params = {
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12), 
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
            'gamma': trial.suggest_float('gamma', 1e-2, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 100.0, log=True), 
            'tree_method': 'hist',
            'device': f'cuda:{gpu_id}',
            'n_jobs': 1
        }        
        
        param = {**fixed_params, **tune_params}

        model = xgb.XGBClassifier(**param)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='neg_log_loss', n_jobs=1)
        
        return scores.mean()
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        raise


def gen_performance_metrics(y_true, y_preds, probabilities, model_name, round_to=3):
    try:
        accuracy = round(accuracy_score(y_true, y_preds), round_to)
        f1 = round(f1_score(y_true, y_preds), round_to)
        recall = round(recall_score(y_true, y_preds), round_to)
        roc_auc = round(roc_auc_score(y_true, probabilities), round_to)
        precision = round(precision_score(y_true, y_preds), round_to)
        tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
        specificity = round(tn / (tn + fp), round_to) if (tn + fp) > 0 else 0
        ppv = precision
        npv = round(tn / (tn + fn), round_to) if (tn + fn) > 0 else 0
        metrics = pd.DataFrame([[model_name, accuracy, f1, recall, specificity, roc_auc, precision, ppv, npv]], 
                            columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Specificity', 'ROC AUC Score', 'Precision', 'PPV', 'NPV'])
        
        logger.info(f"Model performance metrics:\n{metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error generating performance metrics: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"Starting XGBoost 90-day mortality prediction model training on {NUM_GPUS}x GPUs at {start_time}")
        
        X_train, y_train, X_test, y_test, weight_pos_scale = load_data()
        
        fixed_params = {
            "verbosity": 0,
            "scale_pos_weight": weight_pos_scale,
            "random_state": SEED,
            "booster": "gbtree",
            "objective": "binary:logistic",
        }
        
        logger.info(f"Starting Optuna optimization across {NUM_GPUS} GPUs")
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
        study.optimize(objective, n_trials=300, n_jobs=NUM_GPUS, show_progress_bar=True)
        
        logger.info(f"Number of trials conducted: {len(study.trials)}")
        logger.info(f"Best trial value (negative log loss): {study.best_trial.value}")
        
        best_trial = study.best_trial
        logger.info(f"Best trial parameters: {best_trial.params}")
                    
        full_params = {**fixed_params, **best_trial.params}
        full_params['device'] = 'cuda:0'
        full_params['tree_method'] = 'hist'
        
        params_json_path = os.path.join(output_dir, 'best_trial_params.json')
        with open(params_json_path, 'w') as f:
            json.dump(full_params, f, indent=2)
            
        logger.info(f"Best trial parameters saved to {params_json_path}")
        
        logger.info("Training final model with optimized parameters on full training set")
        
        best_model = xgb.XGBClassifier(**full_params)
        
        best_model.fit(X_train, y_train)
        
        logger.info("Final model training completed")
        
        model_path = os.path.join(output_dir, model_filename)
        dump(best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        y_preds = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        
        model_name = 'xgb_28d_nofs'
        metrics_df = gen_performance_metrics(y_test, y_preds, probabilities, model_name, round_to=3)
        metrics_df.to_csv(os.path.join(output_dir, f'metrics_{model_name}.csv'), index=False)
        
        logger.info(f"Number of selected features: {len(features)}")
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"Script completed successfully in {execution_time}")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
