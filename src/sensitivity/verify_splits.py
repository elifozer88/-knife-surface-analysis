# =============================================================================
# verify_splits.py (ALIGNED WITH TIMES NEW ROMAN ACADEMIC FONT STYLE)
# Knife Surface Analysis — Direct Verification for 70/30 and 90/10 splits
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 100)
print("🔍 RUNNING DIRECT SPLIT COMPARISON & GENERATING ACADEMIC HEATMAPS...")
print("=" * 100)

# --- ✍️ AKADEMİK YAZI TİPİ YAPILANDIRMASI (TIMES NEW ROMAN) ---
# Matplotlib'in tüm yazı sistemini serif (Times New Roman) olarak ayarlıyoruz
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Configure output directory directly to sensitivity analysis folder
output_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. LOAD PREPROCESSED DATASETS AND METADATA
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. Please run 'data_prep.py' first."
    )

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

feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()
orig_indices = [i for i, name in enumerate(feature_names) if "sobel" not in name.lower() and "dft" not in name.lower()]
all_indices = list(range(X_all.shape[1]))

paper_metrics = {
    "70_30": {"J48_Un_Orig": 0.470, "J48_Pr_Orig": 0.550, "J48_Un_All": 0.530, "J48_Pr_All": 0.510},
    "90_10": {"J48_Un_Orig": 0.460, "J48_Pr_Orig": 0.580, "J48_Un_All": 0.640, "J48_Pr_All": 0.580}
}

split_configs = [
    {"name": "70_30_Split", "label": "70/30 Split", "test_size": 0.30, "paper_key": "70_30", "seed": 103},
    {"name": "90_10_Split", "label": "90/10 Split", "test_size": 0.10, "paper_key": "90_10", "seed": 204}
]

comparison_rows = []

# =============================================================================
# 2. RUN PIPELINE AND PLOT CUSTOM DUAL-ANNOTATION MATRICES
# =============================================================================
for config in split_configs:
    gss = GroupShuffleSplit(n_splits=1, test_size=config["test_size"], random_state=config["seed"])
    train_idx, test_idx = next(gss.split(X_all, y_clf_3class, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_clf_3class[train_idx], y_clf_3class[test_idx]
    
    # --- SCENARIO A: ORIGINAL FEATURES ---
    X_tr_orig, X_te_orig = X_train[:, orig_indices], X_test[:, orig_indices]
    c45_un_orig = DecisionTreeClassifier(criterion="entropy", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)
    our_un_orig = accuracy_score(y_test, c45_un_orig.predict(X_te_orig))
    
    c45_pr_orig = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)
    our_pr_orig = accuracy_score(y_test, c45_pr_orig.predict(X_te_orig))
    
    # --- SCENARIO B: ALL FILTERS (CHAMPION COHORT) ---
    X_tr_all, X_te_all = X_train[:, all_indices], X_test[:, all_indices]
    c45_un_all = DecisionTreeClassifier(criterion="entropy", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)
    our_un_all = accuracy_score(y_test, c45_un_all.predict(X_te_all))
    
    c45_pr_all = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)
    y_pred_pr_all = c45_pr_all.predict(X_te_all)
    our_pr_all = accuracy_score(y_test, y_pred_pr_all)
    
    # --- 📊 TAM İSTEDİĞİN SADE VE TIMES NEW ROMAN MATRİS ---
    cm = confusion_matrix(y_test, y_pred_pr_all)
            
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=['Class 0\n(Ra < 0.11 µm)', 'Class 1\n(0.11 <= Ra <= 0.45 µm)', 'Class 2\n(Ra > 0.45 µm)'],
        yticklabels=['Class 0\n(Ra < 0.11 µm)', 'Class 1\n(0.11 <= Ra <= 0.45 µm)', 'Class 2\n(Ra > 0.45 µm)'],
        annot_kws={"fontsize": 14, "weight": "normal"}, # Times New Roman ile normal ağırlık çok daha zarif durur
        ax=ax
    )
    
    ax.set_title(f"Confusion Matrix: C4.5 Pruned All Filters ({config['label']})", fontsize=13, fontweight='bold', pad=18)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=12, fontweight='bold')
    ax.set_ylabel("True Class", fontsize=11, labelpad=12, fontweight='bold')
    
    plt.tight_layout()
    
    matrix_path = os.path.join(output_dir, f"matrix_sensitivity_{config['name']}.png")
    plt.savefig(matrix_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ Elegant serif-style heatmap saved successfully to: '{matrix_path}'")

    p_ref = paper_metrics[config["paper_key"]]
    
    comparison_rows.extend([
        {"Split_Type": config["label"], "Model_Config": "C4.5 Unpruned (Original)", "Paper_Accuracy": p_ref["J48_Un_Orig"], "Our_Accuracy": our_un_orig, "Delta": our_un_orig - p_ref["J48_Un_Orig"]},
        {"Split_Type": config["label"], "Model_Config": "C4.5 Pruned (Original)", "Paper_Accuracy": p_ref["J48_Pr_Orig"], "Our_Accuracy": our_pr_orig, "Delta": our_pr_orig - p_ref["J48_Pr_Orig"]},
        {"Split_Type": config["label"], "Model_Config": "C4.5 Unpruned (All Filters)", "Paper_Accuracy": p_ref["J48_Un_All"], "Our_Accuracy": our_un_all, "Delta": our_un_all - p_ref["J48_Un_All"]},
        {"Split_Type": config["label"], "Model_Config": "C4.5 Pruned (All Filters)", "Paper_Accuracy": p_ref["J48_Pr_All"], "Our_Accuracy": our_pr_all, "Delta": our_pr_all - p_ref["J48_Pr_All"]}
    ])

df_comp = pd.DataFrame(comparison_rows)

print("\n" + "=" * 100)
print("📊 DIRECT COMPARISON MATRIX (100% GRAPH ALIGNED)")
print("=" * 100)
print(df_comp.to_string(index=False, formatters={
    'Paper_Accuracy': '{:.2%}'.format, 'Our_Accuracy': '{:.2%}'.format, 'Delta': '{:+.2%}'.format
}))
print("=" * 100)    