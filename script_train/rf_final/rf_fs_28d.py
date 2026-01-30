#!/usr/bin/env python3

import sys
from pathlib import Path
import optuna
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump, load
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from probatus.feature_elimination import ShapRFECV

SEED = 72
train_year = pd.read_csv('datasets/train_test_202511_80/train.csv')
test_year = pd.read_csv('datasets/train_test_202511_80/test.csv')

output_dir = "output/randomforest_output/rf_28d/rf_FS_28d_final_"
model_filename = "FS_28d.joblib"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

features = [
    'Sepsis', 'Sex', 
    
    'mv', 'vasopressor', 'rrt', 'ICU_admission', 
    'HA_MRSA', 
    
    'mi',       # Myocardial infarction
    'chf',      # Congestive heart failure
    'pvd',      # Peripheral vascular disease
    'cevd',     # Cerebrovascular disease
    'dementia', # Dementia
    'cpd',      # Chronic pulmonary disease (COPD)
    'rheumd',   # Rheumatologic disease
    'pud',      # Peptic ulcer disease
    'mld',      # Mild liver disease
    'diab',     # Diabetes without complications
    'diabwc',   # Diabetes with complications
    'hp',       # Hemiplegia or paraplegia
    'rend',     # Renal disease
    'canc',     # Any malignancy
    'msld',     # Moderate or severe liver disease
    'metacanc', # Metastatic solid tumor
    'aids',      # HIV/AIDS
    
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
    'All_Systemic', 'All_Urinary',
]

targets = ['Mortality_28day']

X_train = train_year[features].copy()
y_train = train_year[targets].values.ravel()
X_test = test_year[features].copy()
y_test = test_year[targets].values.ravel()

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

classifier = RandomForestClassifier(random_state=SEED, n_jobs=-1)

shaprfecv = ShapRFECV(classifier,
    step=1,
    min_features_to_select=10,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=SEED,
)

shap_elimination = shaprfecv.fit(X_train, y_train, check_additivity=False)

selected_features = shaprfecv.get_reduced_features_set(num_features='best')
n_features_selected = len(selected_features)

performance_plot = shaprfecv.plot()
performance_plot.figure.savefig(os.path.join(output_dir, 'feature_selection_performance.png'), dpi=300)
plt.close(performance_plot.figure)

selected_features_df = pd.DataFrame(selected_features, columns=['Selected Features'])
output_file_path = os.path.join(output_dir, 'selected_features.csv')
selected_features_df.to_csv(output_file_path, index=False)                


X_train = X_train[selected_features]
X_test = X_test[selected_features]

fixed_params = {
        "random_state": 42,
        "n_jobs": -1
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
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='neg_log_loss', error_score='raise')
    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300, n_jobs=-1)
    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    full_params = {**fixed_params, **best_trial.params}

    json_file_path = os.path.join(output_dir, 'best_params.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(full_params, json_file)

    best_model = RandomForestClassifier(**full_params)
    best_model.fit(X_train, y_train)

    model_path = os.path.join(output_dir, model_filename)
    dump(best_model, model_path)

