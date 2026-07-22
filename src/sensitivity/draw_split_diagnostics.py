# =============================================================================
# draw_split_diagnostic.py
# Sensitivity Analysis — Model Accuracy vs. Training Data Volume (Fig 4 Replication)
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

print("=" * 75)
print("GENERATING EXACT SENSITIVITY CURVES MATCHING PAPER FIGURE 4 MAP...")
print("=" * 75)

# -----------------------------------------------------------------------------
# DIRECTORY CONFIGURATION (TARGET: outputs/figures/4_sensitivity_analysis/)
# -----------------------------------------------------------------------------
output_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. ACQUIRE PROCESSED DATASETS AND GROUP IDS
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. "
        f"Please run the preprocessing script ('data_prep.py') before executing sensitivity runs."
    )

data = np.load(data_path, allow_pickle=True)
X_all = np.vstack([data["X_train"], data["X_test"]])
y_clf_3class = np.concatenate([data["y_clf_train"], data["y_clf_test"]])

# Reconstruct consistent group identifiers to avoid target leakage during group splitting
if "train_groups" in data:
    train_groups = data["train_groups"]
    test_groups = np.array([f"test_grp_{i}" for i in range(len(data["X_test"]))], dtype=object)
    train_groups_str = np.array([f"train_grp_{g}" for g in train_groups], dtype=object)
    groups_all = np.concatenate([train_groups_str, test_groups]).astype(str)
else:
    groups_all = np.array([f"grp_{i}" for i in range(len(X_all))]).astype(str)

feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# Define feature spaces (Pure structural variables vs. Complete filter features)
orig_indices = [i for i, name in enumerate(feature_names) if "sobel" not in name.lower() and "dft" not in name.lower()]
all_indices = list(range(X_all.shape[1]))

# Configure sequential training sizes matching the paper's methodology
train_percentages = [0.10, 0.30, 0.50, 0.70, 0.90]
results_db = []

# =============================================================================
# 2. SYSTEMATIC MODEL TRAINING PIPELINE (SCALING SIMULATION)
# =============================================================================
for p in train_percentages:
    test_size = 1.0 - p
    
    # Initialize Group-based split to ensure strict physical isolation of knife specimens
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_clf_3class, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_clf_3class[train_idx], y_clf_3class[test_idx]
    
    # --- SCENARIO A: ORIGINAL STRUCTURAL FEATURES ONLY ---
    X_tr_orig, X_te_orig = X_train[:, orig_indices], X_test[:, orig_indices]
    
    j48_un_orig = DecisionTreeClassifier(criterion="entropy", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)
    j48_pr_orig = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)
    cart_un_orig = DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)
    cart_pr_orig = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_orig, y_train)

    # --- SCENARIO B: ALL IMAGING FILTERS INCLUDED (SOBEL & DFT) ---
    X_tr_all, X_te_all = X_train[:, all_indices], X_test[:, all_indices]
    
    j48_un_all = DecisionTreeClassifier(criterion="entropy", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)
    j48_pr_all = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)
    cart_un_all = DecisionTreeClassifier(criterion="gini", max_depth=12, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)
    cart_pr_all = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.001, class_weight="balanced", random_state=42).fit(X_tr_all, y_train)

    # Evaluate predictive accuracy across all simulated models
    results_db.append({
        "Train_Percentage": p * 100,  # Saved as 10.0, 30.0, 50.0, etc.
        "J48_Un_Orig": accuracy_score(y_test, j48_un_orig.predict(X_te_orig)),
        "J48_Pr_Orig": accuracy_score(y_test, j48_pr_orig.predict(X_te_orig)),
        "CART_Un_Orig": accuracy_score(y_test, cart_un_orig.predict(X_te_orig)),
        "CART_Pr_Orig": accuracy_score(y_test, cart_pr_orig.predict(X_te_orig)),
        "J48_Un_All": accuracy_score(y_test, j48_un_all.predict(X_te_all)),
        "J48_Pr_All": accuracy_score(y_test, j48_pr_all.predict(X_te_all)),
        "CART_Un_All": accuracy_score(y_test, cart_un_all.predict(X_te_all)),
        "CART_Pr_All": accuracy_score(y_test, cart_pr_all.predict(X_te_all)),
    })

df_res = pd.DataFrame(results_db)

# =============================================================================
# 3. REPLICATING PAPER FIGURE 4 GRAPHICAL DESIGN STANDARDS
# =============================================================================
plt.figure(figsize=(13, 7), dpi=300)
ax = plt.subplot(111)

# Apply formal academic styles: Clean white background with subtle gray grids
ax.set_facecolor('white')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.grid(True, linestyle='-', color='#e0e0e0', linewidth=0.8)

# Blue/Navy Series: Original Feature Subspace
ax.plot(df_res["Train_Percentage"], df_res["J48_Un_Orig"], marker='^', markersize=7, linestyle='-.', color='#1f3a60', linewidth=2, label="J48 unpruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["J48_Pr_Orig"], marker='s', markersize=7, linestyle=':', color='#5dade2', linewidth=2, label="J48 pruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Un_Orig"], marker='o', markersize=7, linestyle='-', color='#3498db', linewidth=2, label="CART unpruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Pr_Orig"], marker='d', markersize=7, linestyle='--', color='#2980b9', linewidth=2, label="CART pruned original 3 classes")

# Green/Forest Series: All Filters Feature Space (Advanced Preprocessed)
ax.plot(df_res["Train_Percentage"], df_res["J48_Un_All"], marker='^', markersize=7, linestyle='-.', color='#1e824c', linewidth=2, label="J48 unpruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["J48_Pr_All"], marker='s', markersize=7, linestyle=':', color='#a2d9ce', linewidth=2, label="J48 pruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Un_All"], marker='o', markersize=7, linestyle='-', color='#27ae60', linewidth=2, label="CART unpruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Pr_All"], marker='d', markersize=7, linestyle='--', color='#2ecc71', linewidth=2, label="CART pruned all filters 3 classes")

# Set formal academic titles and axis labels
ax.set_title("Correctly Classified Data in Dependence of the Quantity of Training Data", fontsize=12, pad=15, fontweight='bold')
ax.set_xlabel("Percentage of Training Data (%)", fontsize=11, labelpad=8)
ax.set_ylabel("Correctly Predicted Data (Accuracy)", fontsize=11, labelpad=8)

# GÜNCELLEME: Sınır kilitleri artık tam sayı (10-100) formatıyla uyumlu!
ax.set_ylim(0.30, 0.65)
ax.set_xlim(5, 95)
ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Configure percentage formatters
ax.xaxis.set_major_formatter(mtick.PercentFormatter(100.0, decimals=0)) # Handles 10.0 to 100.0 correctly
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))  # Handles 0.30 to 0.65 accuracy values

# Position legend on the right side in a vertical block structure
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="white", edgecolor="#bdc3c7", fontsize=10)

# 4. EXPORT PLOT TO SERIALIZED WORKSPACE
plt.tight_layout()
output_path = os.path.join(output_dir, "Fig8_paper_style_sensitivity_curves.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Replicated paper style sensitivity curve saved successfully to: '{output_path}'")