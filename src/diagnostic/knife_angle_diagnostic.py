# =============================================================================
# knife_angle_diagnostic.py (HYBRID VERSION - 100% CLEAN & ALIGNED)
# Analysing Shared Failures Across Different Knife Angles (Linie)
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit

print("=" * 100)
print("🎯 RUNNING SAFE ANGLE-BASED (LINIE) DIAGNOSTIC...")
print("=" * 100)

# 1. DYNAMIC FILE DETECTION AND LOADING
data_path = "data/processed/split_data.npz"
raw_data_dir = "data/raw" # Scanning your raw folder
raw_data_path = None

if not os.path.exists(data_path):
    raise FileNotFoundError(f"ERROR: '{data_path}' not found. Please run 'data_prep.py' first.")

# Find the Kochmesser Excel or CSV in data/raw
for file in os.listdir(raw_data_dir):
    if "kochmesser" in file.lower() and (file.endswith(".csv") or file.endswith(".xlsx")):
        raw_data_path = os.path.join(raw_data_dir, file)
        break

if not raw_data_path:
    # Fallback to any file in raw directory
    all_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv") or f.endswith(".xlsx")]
    if all_files:
        raw_data_path = os.path.join(raw_data_dir, all_files[0])
    else:
        raise FileNotFoundError(f"ERROR: No dataset file found inside '{raw_data_dir}/' directory.")

print(f"[INFO] Raw metadata file detected: {raw_data_path}")

# Load raw file dynamically to extract Name and Linie
if raw_data_path.endswith(".csv"):
    df_raw = pd.read_csv(raw_data_path)
elif raw_data_path.endswith(".xlsx"):
    try:
        df_raw = pd.read_excel(raw_data_path)
    except ImportError:
        print("[INFO] Installing 'openpyxl' for Excel support...")
        os.system("pip install openpyxl")
        df_raw = pd.read_excel(raw_data_path)

# Extract matching Name and Linie
knife_names_all = df_raw['Name'].values
lines_all = df_raw['Linie'].values

# Load clean, NaN-free classification features and targets from NPZ
data = np.load(data_path, allow_pickle=True)
X_all = np.vstack([data["X_train"], data["X_test"]])
y_clf_3class = np.concatenate([data["y_clf_train"], data["y_clf_test"]])

if "train_groups" in data:
    train_groups = data["train_groups"]
    test_groups = np.array([f"test_grp_{i}" for i in range(len(data["X_test"]))], dtype=object)
    train_groups_str = np.array([f"train_grp_{g}" for g in train_groups], dtype=object)
    groups_all = np.concatenate([train_groups_str, test_groups]).astype(str)
else:
    groups_all = np.array([f"grp_{i}" for i in range(len(X_all))]).astype(str)

# Define split configurations (100% matched with verify_splits.py)
# Define split configurations (Now including 80/20 for full coverage!)
split_configs = [
    {"name": "70_30_split", "label": "70/30 Split", "test_size": 0.30, "seed": 103},
    {"name": "80_20_split", "label": "80/20 Split", "test_size": 0.20, "seed": 42}, # Eklendi 🎯 (Genel random_state'imizle uyumlu seed)
    {"name": "90_10_split", "label": "90/10 Split", "test_size": 0.10, "seed": 204}
]

models = {
    "unpruned": {
        "cart": DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced", random_state=42),
        "c45": DecisionTreeClassifier(criterion="entropy", max_depth=12, class_weight="balanced", random_state=42)
    },
    "pruned": {
        "cart": DecisionTreeClassifier(criterion="gini", ccp_alpha=0.001, class_weight="balanced", random_state=42),
        "c45": DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42)
    }
}

# 2. RUN DIAGNOSTIC PIPELINE
for config in split_configs:
    print(f"\nProcessing {config['label']}...")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=config["seed"])
    train_idx, test_idx = next(gss.split(X_all, y_clf_3class, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_clf_3class[train_idx], y_clf_3class[test_idx]
    
    test_knives = knife_names_all[test_idx]
    test_lines = lines_all[test_idx]
    
    for prune_name, model_set in models.items():
        cart_model = model_set["cart"].fit(X_train, y_train)
        c45_model = model_set["c45"].fit(X_train, y_train)
        
        cart_preds = cart_model.predict(X_test)
        c45_preds = c45_model.predict(X_test)
        
        diagnostics_df = pd.DataFrame({
            'Knife_ID': test_knives,
            'Linie_Angle': test_lines,
            'Actual_Class': y_test,
            'CART_Prediction': cart_preds,
            'C45_Prediction': c45_preds
        })
        
        diagnostics_df['CART_Correct'] = diagnostics_df['CART_Prediction'] == diagnostics_df['Actual_Class']
        diagnostics_df['C45_Correct'] = diagnostics_df['C45_Prediction'] == diagnostics_df['Actual_Class']
        
        shared_failures = diagnostics_df[~diagnostics_df['CART_Correct'] & ~diagnostics_df['C45_Correct']].copy()
        perfect_mimics = diagnostics_df[diagnostics_df['CART_Correct'] & diagnostics_df['C45_Correct']].copy()
        
        target_dir = os.path.join("outputs", config["name"], prune_name)
        os.makedirs(target_dir, exist_ok=True)
        
        diagnostics_df.to_csv(os.path.join(target_dir, "test_knives_behavior_analysis.csv"), index=False)
        shared_failures.to_csv(os.path.join(target_dir, "always_incorrect_knives.csv"), index=False)
        perfect_mimics.to_csv(os.path.join(target_dir, "always_correct_knives.csv"), index=False)
        
        if len(shared_failures) > 0:
            angle_error_counts = shared_failures['Linie_Angle'].value_counts().sort_index()
            angle_report = pd.DataFrame({
                'Linie_Angle': angle_error_counts.index,
                'Error_Count': angle_error_counts.values,
                'Percentage_of_Total_Shared_Errors': np.round((angle_error_counts.values / len(shared_failures)) * 100, 2)
            })
            angle_report_path = os.path.join(target_dir, "angle_error_distribution.csv")
            angle_report.to_csv(angle_report_path, index=False)
            print(f"  -> [{prune_name.upper()}] Angle analysis saved to: {angle_report_path}")

print("\n" + "=" * 100)
print("✅ DIAGNOSTIC ANALYSIS FINISHED SUCCESSFULLY!")
print("=" * 100 + "\n")