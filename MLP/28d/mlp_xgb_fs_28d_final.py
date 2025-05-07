#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score,RepeatedStratifiedKFold
from sklearn import metrics
from joblib import dump
import optuna
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import sys

SEED = 42

train_year = pd.read_csv('datasets/train_test_28d/train_df_28d.csv')
test_year = pd.read_csv('datasets/train_test_28d/test_df_28d.csv')
features_csv_path = 'output/xgb/XGBoost_outputs_28d/28d_shaprfecv_final/selected_features.csv' 

output_dir = "output/mlp_xgb/mlp_28day/mlp_fs_28d"

model_filename = "mlp_28d_fs_xgb.joblib"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_selected_features(features_csv_path):
    features_df = pd.read_csv(features_csv_path)
    features = features_df.iloc[:, 0].tolist()
    return features

features = get_selected_features(features_csv_path)

targets = ['Mortality_28day']

X_train = train_year[features].copy()
y_train = train_year[targets].values.ravel()
X_test = test_year[features].copy()
y_test = test_year[targets].values.ravel()

continuous_features = [
    'Age','CCI',
    'hospitalised_until_BSI',
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

def gen_performance_metrics(y_true, y_preds, probabilities, model_name, round_to=3):
    accuracy = round(metrics.accuracy_score(y_true, y_preds), round_to)
    f1_score = round(metrics.f1_score(y_true, y_preds), round_to)
    recall = round(metrics.recall_score(y_true, y_preds), round_to)
    roc_auc = round(metrics.roc_auc_score(y_true, probabilities), round_to)
    precision = round(metrics.precision_score(y_true, y_preds), round_to)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_preds).ravel()
    specificity = round(tn / (tn + fp), round_to) if (tn + fp) > 0 else 0

    return pd.DataFrame([[model_name, accuracy, f1_score, recall, specificity, roc_auc, precision]], 
                        columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Specificity', 'ROC AUC Score', 'Precision'])

fixed_params = {
        "random_state": SEED,
        "max_iter": 5000
        }

def objective(trial):
    hidden_layer_options = ["50", "100", "50,50"]
    hidden_layer_choice = trial.suggest_categorical("hidden_layer_sizes", hidden_layer_options)
    
    if "," in hidden_layer_choice:
        hidden_layer_sizes = tuple(map(int, hidden_layer_choice.split(",")))
    else:
        hidden_layer_sizes = (int(hidden_layer_choice),)
        
    activation = trial.suggest_categorical("activation", ['relu', 'tanh'])
    solver = trial.suggest_categorical("solver", ['adam', 'sgd'])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])
    
    tune_params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "learning_rate": learning_rate,
    }
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features)
        ],
        remainder='passthrough'
    )

    params = {**fixed_params, **tune_params}
    model = MLPClassifier(**params)

    pipeline_optuna = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('to_dataframe', DataFrameConverter(columns=features)),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=SEED)),
        ('classifier', model)
    ])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline_optuna, X_train, y_train, cv=skf, scoring='neg_log_loss', error_score='raise')
    return scores.mean()

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)
    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_params = best_trial.params
    hidden_layer_choice = best_params["hidden_layer_sizes"]
    if isinstance(hidden_layer_choice, str):
        if "," in hidden_layer_choice:
            best_params["hidden_layer_sizes"] = tuple(map(int, hidden_layer_choice.split(",")))
        else:
            best_params["hidden_layer_sizes"] = (int(hidden_layer_choice),)

    full_params = {**fixed_params, **best_params}

    best_model = MLPClassifier(**full_params)

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

    model_name = 'mlp_nopara_threshold_unadjusted'
    metrics_df = gen_performance_metrics(y_test, y_preds, probabilities, model_name, round_to=3)
    pd.set_option('display.max_rows', None)
    
    print(metrics_df)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_mlp_nopara_threshold_unadjusted.csv'), index=False)



