import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12
})

output_dir = 'SHAP/90d_noweight'
os.makedirs(output_dir, exist_ok=True)

test_data = pd.read_csv('datasets/train_test_202511_80/test.csv')

model_filename = 'all_final_model/xgb_fs/xgb_fs_90d_noweight.joblib'
assert os.path.exists(model_filename), "Model file not found."
model = joblib.load(model_filename)

def get_selected_features(model):
    features = list(model.feature_names_in_)
    return features

features = get_selected_features(model)

missing_features = [feature for feature in features if feature not in test_data.columns]
if missing_features:
    raise ValueError(f"The following features are missing from the test dataset: {missing_features}")

X_test = test_data[features]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

feature_name_mapping = {
    "Sepsis": "Sepsis (Yes=1; No=0)",
    "Sex": "Sex (Male=0; Female=1)",
    "Age": "Age (Years)",
    "mv": "Mechanical Ventilation (Yes=1; No=0)",
    "vasopressor": "Vasopressor Use (Yes=1; No=0)",
    "rrt": "Renal Replacement Therapy (Yes=1; No=0)",
    "ICU_admission": "ICU Admission (Yes=1; No=0)",
    "HA_MRSA": "Hospital-Acquired MRSA (Yes=1; No=0)",
    "delta SOFA score": "ΔSOFA Score",
    "prehosp_SOFA_total": "Pre-hospital Total SOFA Score",
    "hospital_SOFA_total": "SOFA Score at the Time of Infection",
    "CCI": "Charlson Comorbidity Index",
    "mi": "Myocardial Infarction (Yes=1; No=0)",
    "chf": "Congestive Heart Failure (Yes=1; No=0)",
    "pvd": "Peripheral Vascular Disease (Yes=1; No=0)",
    "cevd": "Cerebrovascular Disease (Yes=1; No=0)",
    "dementia": "Dementia (Yes=1; No=0)",
    "cpd": "Chronic Pulmonary Disease (Yes=1; No=0)",
    "rheumd": "Rheumatic Disease (Yes=1; No=0)",
    "pud": "Peptic Ulcer Disease (Yes=1; No=0)",
    "mld": "Mild Liver Disease (Yes=1; No=0)",
    "diab": "Diabetes without Complications (Yes=1; No=0)",
    "diabwc": "Diabetes with Complications (Yes=1; No=0)",
    "hp": "Hemiplegia or Paraplegia (Yes=1; No=0)",
    "rend": "Renal Disease (Yes=1; No=0)",
    "canc": "Cancer (Yes=1; No=0)",
    "msld": "Moderate or Severe Liver Disease (Yes=1; No=0)",
    "metacanc": "Metastatic Cancer (Yes=1; No=0)",
    "df_bili_Lab Result of Last Timepoint": "Bilirubin (μmol/L) of Last Time Point",
    "df_bili_Lab Result of First Timepoint": "Bilirubin (μmol/L) of First Time Point",
    "df_crea_Lab Result of Last Timepoint": "Creatinine (μmol/L) of Last Time Point",
    "df_crea_Lab Result of First Timepoint": "Creatinine (μmol/L) of First Time Point",
    "df_plate_Lab Result of Last Timepoint": "Platelet Count (x 10³/µL) of Last Time Point",
    "df_plate_Lab Result of First Timepoint": "Platelet Count (x 10³/µL) of First Time Point",
    "df_plate_Last - First": "Platelet Count (Last - First) (x 10³/µL)",
    "df_crea_Last - First": "Creatinine (Last - First) (μmol/L)",
    "df_bili_Last - First": "Bilirubin (Last - First) (μmol/L)",
    "MRSA_Lower respiratory": "MRSA (+) Lower Respiratory Sites (Yes/No)",
    "MRSA_Musculoskeletal": "MRSA (+) Musculoskeletal Sites (Yes/No)",
    "MRSA_Other_unknown_sites": "MRSA (+) Other/Unknown Sites (Yes/No)",
    "MRSA_Prosthesis/Lines": "MRSA (+) Prosthesis/Lines Sites (Yes/No)",
    "MRSA_Skin/Wound": "MRSA (+) Skin/Wound Sites (Yes/No)",
    "MRSA_Urinary": "MRSA (+) Urinary Sites (Yes/No)",
    "All_Gastrointestinal/Peritoneal": "Other Gastrointestinal Pathogen (+) (Yes/No)",
    "All_Lower respiratory": "Other Lower Respiratory Pathogen (+) (Yes/No)",
    "All_Skin/Wound": "Other Skin/Wound Pathogen (+) (Yes/No)",
    "All_Systemic": "Other Systemic Pathogen (+) (Yes/No)",
    "All_Urinary": "Other Urinary Pathogen (+) (Yes/No)",
}
adjusted_feature_names = [feature_name_mapping.get(feature, feature) for feature in features]

shap_values_combined = shap_values

custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom", 
    ["#ACD6EC", "gray", "#F5A889"]
)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_combined,
    X_test, 
    feature_names=adjusted_feature_names,  
    plot_type="dot",
    max_display=X_test.shape[1],
    cmap=custom_cmap,
    show=False
)

plt.savefig(os.path.join(output_dir, 'shap_90d_xgb.tiff'), bbox_inches='tight', pad_inches=0.3, dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
shap.summary_plot(
    shap_values_combined,
    X_test, 
    feature_names=adjusted_feature_names,  
    plot_type="dot",
    max_display=15,
    cmap=custom_cmap,
    show=False
)

plt.savefig(os.path.join(output_dir, 'shap_90d_15_xgb.tiff'), bbox_inches='tight', pad_inches=0.3, dpi=300)
plt.close()
