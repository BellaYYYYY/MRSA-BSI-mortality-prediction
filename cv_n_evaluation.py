import xgboost as xgb

import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
    precision_score, brier_score_loss, confusion_matrix
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from joblib import load
import warnings
from xgboost import Booster
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert numpy array to DataFrame with the specified columns
        return pd.DataFrame(X, columns=self.columns)

def gen_performance_metrics(y_true, y_preds, probabilities, model_name, round_to=3):
    accuracy = round(accuracy_score(y_true, y_preds), round_to)
    f1 = round(f1_score(y_true, y_preds), round_to)
    recall = round(recall_score(y_true, y_preds), round_to)
    roc_auc = round(roc_auc_score(y_true, probabilities), round_to)
    precision = round(precision_score(y_true, y_preds), round_to)
    brier = round(brier_score_loss(y_true, probabilities), round_to)
    specificity = round(tn / (tn + fp), round_to) if (tn + fp) > 0 else 0

    return pd.DataFrame([[model_name, accuracy, f1, recall, specificity, roc_auc, precision, brier]],
                        columns=['Model', 'Accuracy', 'F1 Score', 'Recall', 'Specificity',
                                 'ROC AUC Score', 'Precision', 'Brier Score'])

def get_model_features(model, X_test):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        return model.get_booster().feature_names
    else:
        return list(X_test.columns)

# Function to load XGBoost models safely
def load_model_safely(model_path, model_type):
    if model_type == "xgb":
        try:
            # Try loading the model using XGBoost's Booster class
            model = Booster()
            model.load_model(model_path)
            print(f"Loaded XGBoost model using Booster from {model_path}")
        except Exception:
            # Fallback to joblib if Booster fails
            model = load(model_path)
            print(f"Loaded XGBoost model using joblib from {model_path}")
    else:
        # For non-XGBoost models, use joblib
        model = load(model_path)
    return model

datasets_models = {
    
    "90d": {
        "train_data_path": "./datasets/train_test_90d/train_df_90d.csv",
        "test_data_path": "./datasets/train_test_90d/test_df_90d.csv",
        "target_column": "Mortality_90day",
        "models": {
            "xgb": "output/xgb/XGBoost_outputs_90d/90d_shaprfecv_final/xgb_shaprfecv_90d.joblib",
            "rf": "output/randomforest_output/rf_90d/rf_FS_90d_final_step1/FS_90d.joblib",
            "mlp": "output/mlp_xgb/mlp_90day/mlp_fs_90d/mlp_90d_fs_xgb.joblib",
        }
    },
    "28d": {
        "train_data_path": "./datasets/train_test_28d/train_df_28d.csv",
        "test_data_path": "./datasets/train_test_28d/test_df_28d.csv",
        "target_column": "Mortality_28day",
        "models": {
            "xgb": "output/xgb/XGBoost_outputs_28d/28d_shaprfecv_final/xgb_shaprfecv_28d.joblib",
            "rf": "output/randomforest_output/rf_28d/rf_FS_28d_final_step1/FS_28d.joblib",
            "mlp": "output/mlp_xgb/mlp_28day/mlp_fs_28d/mlp_28d_fs_xgb.joblib",
        }
    },
    "1year": {
        "train_data_path": "./datasets/train_test_1y/train_df_1y.csv",
        "test_data_path": "./datasets/train_test_1y/test_df_1y.csv",
        "target_column": "Mortality_1year",
        "models": {
            "xgb": "output/xgb/XGBoost_outputs_1y/1y_shaprfecv_final/xgb_shaprfecv_1y.joblib",
            "rf": "output/randomforest_output/rf_1y/rf_FS_1y_final_step1/FS_1y.joblib",
            "mlp": "output/mlp_xgb/mlp_1year/mlp_fs_1y/mlp_1y_fs_xgb.joblib",
        }
    },
    "hospital": {
        "train_data_path": "./datasets/train_test_hospital/train_df_hospital.csv",
        "test_data_path": "./datasets/train_test_hospital/test_df_hospital.csv",
        "target_column": "Episode_mortality_allcause",
        "models": {
            "xgb": "output/xgb/XGBoost_outputs_hospital/hospital_shaprfecv_final/xgb_shaprfecv_hospital.joblib",
            "rf": "output/randomforest_output/rf_hospital/rf_FS_hospital_final_step1/FS_hospital.joblib",
            "mlp": "output/mlp_xgb/mlp_hospital/mlp_fs_hospital/mlp_hospital_fs_xgb.joblib",
        }
    }
}

cv_fold_results = []  

test_results = []

for outcome, paths in datasets_models.items():
    print(f"\nProcessing outcome: {outcome}")
    target_column = paths["target_column"]

    # Load train and test datasets
    train_df = pd.read_csv(paths["train_data_path"])
    test_df = pd.read_csv(paths["test_data_path"])
    y_train = train_df[target_column]
    y_test = test_df[target_column]

    for model_name, model_path in paths["models"].items():
        print(f"\nEvaluating model: {model_name}")

        model = load_model_safely(model_path, model_name)

        model_features = get_model_features(model, train_df)

        X_train = train_df[model_features]
        X_test = test_df[model_features]

        # Cross-validation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        fold_metrics = []

        for fold_idx, (train_index, val_index) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            model_cv = clone(model)
            model_cv.fit(X_tr, y_tr)

            if hasattr(model_cv, "predict_proba"):
                y_val_prob = model_cv.predict_proba(X_val)[:, 1]
            else:
                raise ValueError(f"Model {model_name} does not support predict_proba.")
                
            y_val_pred = (y_val_prob >= 0.5).astype(int)

            metrics_df = gen_performance_metrics(y_val, y_val_pred, y_val_prob, model_name)
            metrics = metrics_df.iloc[0].to_dict()
            metrics["Fold"] = fold_idx
            metrics["Outcome"] = outcome
            fold_metrics.append(metrics)

        cv_fold_results.extend(fold_metrics)


        # Test set evaluation
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Model {model_name} does not support predict_proba.")
        
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        test_metrics_df = gen_performance_metrics(y_test, y_test_pred, y_test_prob, model_name)
        test_metrics = test_metrics_df.iloc[0].to_dict()
        test_metrics["Outcome"] = outcome
        test_results.append(test_metrics)

# Save fold-level CV results to a CSV file
cv_fold_results_df = pd.DataFrame(cv_fold_results)
cv_fold_results_csv_path = "brier/cv_outcomes/cv_fold_metrics.csv"
cv_fold_results_df.to_csv(cv_fold_results_csv_path, index=False)

# Save test set results to another CSV file
test_results_df = pd.DataFrame(test_results)
test_results_csv_path = "brier/cv_outcomes/test_metrics.csv"
test_results_df.to_csv(test_results_csv_path, index=False)

print(f"\nFold-level CV metrics saved to {cv_fold_results_csv_path}")
print(f"Test set metrics saved to {test_results_csv_path}")
