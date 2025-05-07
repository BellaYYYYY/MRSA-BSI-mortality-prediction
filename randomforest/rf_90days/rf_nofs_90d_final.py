#!/usr/bin/env python3
import optuna
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix, roc_curve
from joblib import dump
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import json

SEED = 123

train_year = pd.read_csv('datasets/train_test_90d/train_df_90d.csv')
test_year = pd.read_csv('datasets/train_test_90d/test_df_90d.csv')

output_dir = "output/randomforest_output/rf_90d/rf_noFS_90d_final"
model_filename = "noFS_90d.joblib"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

features = [
    'Sepsis', 'Sex', 'Age', 
    'mv', 'vasopressor', 'rrt', 'ICU_admission', 
    'HA_MRSA', 
    'delta SOFA score', 'prehosp_SOFA_total', 'hospital_SOFA_total', 
    'CCI', 
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


targets = ['Mortality_90dear']

X_train = train_year[features].copy()
y_train = train_year[targets].values.ravel()
X_test = test_year[features].copy()
y_test = test_year[targets].values.ravel()

continuous_features = [
    'Age',
    'CCI',
    'delta SOFA score',
    'prehosp_SOFA_total',
    'hospital_SOFA_total',
    'Number of Surgeries',
    'Number_emergency_surgeries',
    'df_bili_Std Lab Result',
    'df_bili_Lab Result of Last Timepoint',
    'df_bili_Lab Result of First Timepoint',
    'df_crea_Std Lab Result',
    'df_crea_Lab Result of Last Timepoint',
    'df_crea_Lab Result of First Timepoint',
    'df_plate_Std Lab Result',
    'df_plate_Lab Result of Last Timepoint',
    'df_plate_Lab Result of First Timepoint',
]

continuous_features = [feature for feature in continuous_features if feature in features]

class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

fixed_params = {
        "random_state": SEED,
        "n_jobs": -1,
        }
        
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", low=10, high=1000)
    max_features = trial.suggest_categorical("max_features", choices=['sqrt', 'log2', None])
    max_depth = trial.suggest_int("max_depth", low=10, high=110)
    min_samples_split = trial.suggest_int("min_samples_split", low=2, high=10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", low=1, high=5)
    
    tune_params = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf
    }

    param = {**fixed_params, **tune_params}
    model = RandomForestClassifier(**param)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features)
        ],
        remainder='passthrough'
    )

    pipeline_optuna = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('to_dataframe', DataFrameConverter(columns=features)),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', model)
    ])
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline_optuna, X_train, y_train, cv=skf, scoring='neg_log_loss', error_score='raise')
    return scores.mean()

def gen_performance_metrics(y_true, y_preds, probabilities, model_name, round_to=3):
    accuracy = round(metrics.accuracy_score(y_true, y_preds), round_to)
    f1_score = round(metrics.f1_score(y_true, y_preds), round_to)
    recall = round(metrics.recall_score(y_true, y_preds), round_to)
    roc_auc = round(metrics.roc_auc_score(y_true, probabilities), round_to)
    precision = round(metrics.precision_score(y_true, y_preds), round_to)
    brier_score = round(metrics.brier_score_loss(y_true, probabilities), round_to)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_preds).ravel()
    specificity = round(tn / (tn + fp), round_to) if (tn + fp) > 0 else 0

    return pd.DataFrame([[model_name, accuracy, f1_score, recall, specificity, roc_auc, precision, brier_score]], 
                        columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Specificity', 'ROC AUC Score', 'Precision', 'Brier Score'])

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)
    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    full_params = {**fixed_params, **best_trial.params}
    params_path = os.path.join(output_dir, 'best_full_params.json')

    best_model = RandomForestClassifier(**full_params)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features)
        ],
        remainder='passthrough'
    )

    best_pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('to_dataframe', DataFrameConverter(columns=features)),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', best_model)
    ])

    best_pipeline.fit(X_train, y_train)
    model_path = os.path.join(output_dir, model_filename)
    dump(best_pipeline, model_path)

    y_preds = best_pipeline.predict(X_test)
    probabilities = best_pipeline.predict_proba(X_test)[:, 1]

    model_name = 'rf_noFS'
    metrics_df = gen_performance_metrics(y_test, y_preds, probabilities, model_name, round_to=3)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_rf_noFS.csv'), index=False)

