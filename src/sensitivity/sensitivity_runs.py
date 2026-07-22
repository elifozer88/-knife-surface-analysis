# =============================================================================
# sensitivity_runs.py (EXACT HINZ ET AL. 2019 REPLICATION | Y-AXIS: 30% TO 65%)
# Knife Surface Analysis — Exact Paper Method Alignment
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
print("REPLICATING EXACT HINZ ET AL. (2019) SENSITIVITY METHODOLOGY...")
print("=" * 75)

output_dir = "outputs/figures/4_sensitivity_analysis"
os.makedirs(output_dir, exist_ok=True)

# 1. LOAD PREPROCESSED DATASETS
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"CRITICAL ERROR: '{data_path}' not found.")

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

train_percentages = [0.10, 0.30, 0.50, 0.70, 0.90]
results_db = []

# =============================================================================
# 2. EXACT MODEL CALCULATIONS AS IN THE PAPER
# =============================================================================
for p in train_percentages:
    test_size = 1.0 - p
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_clf_3class, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_clf_3class[train_idx], y_clf_3class[test_idx]
    
    # --- Scenario A: Original Features ---
    X_tr_orig, X_te_orig = X_train[:, orig_indices], X_test[:, orig_indices]
    
    # C4.5 (J48) Unpruned & Pruned
    c45_un_orig = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X_tr_orig, y_train)
    c45_pr_orig = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.01, random_state=42).fit(X_tr_orig, y_train)
    
    # CART Unpruned & Pruned
    cart_un_orig = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X_tr_orig, y_train)
    cart_pr_orig = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.01, random_state=42).fit(X_tr_orig, y_train)

    # --- Scenario B: All Filters Features ---
    X_tr_all, X_te_all = X_train[:, all_indices], X_test[:, all_indices]
    
    # C4.5 (J48) Unpruned & Pruned
    c45_un_all = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X_tr_all, y_train)
    c45_pr_all = DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.01, random_state=42).fit(X_tr_all, y_train)
    
    # CART Unpruned & Pruned
    cart_un_all = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X_tr_all, y_train)
    cart_pr_all = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.01, random_state=42).fit(X_tr_all, y_train)

    results_db.append({
        "Train_Percentage": p * 100,
        "C4.5_Un_Orig": accuracy_score(y_test, c45_un_orig.predict(X_te_orig)),
        "C4.5_Pr_Orig": accuracy_score(y_test, c45_pr_orig.predict(X_te_orig)),
        "CART_Un_Orig": accuracy_score(y_test, cart_un_orig.predict(X_te_orig)),
        "CART_Pr_Orig": accuracy_score(y_test, cart_pr_orig.predict(X_te_orig)),
        "C4.5_Un_All": accuracy_score(y_test, c45_un_all.predict(X_te_all)),
        "C4.5_Pr_All": accuracy_score(y_test, c45_pr_all.predict(X_te_all)),
        "CART_Un_All": accuracy_score(y_test, cart_un_all.predict(X_te_all)),
        "CART_Pr_All": accuracy_score(y_test, cart_pr_all.predict(X_te_all)),
    })

df_res = pd.DataFrame(results_db)

# =============================================================================
# 3. PLOTTING FIG. 4 PAPER EXACT STYLE
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)

# Blue Series (Original)
ax.plot(df_res["Train_Percentage"], df_res["C4.5_Un_Orig"], marker='^', markersize=7, linestyle='-.', color='#1f3a60', linewidth=2, label="C4.5 unpruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["C4.5_Pr_Orig"], marker='s', markersize=7, linestyle=':', color='#5dade2', linewidth=2, label="C4.5 pruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Un_Orig"], marker='o', markersize=7, linestyle='-', color='#3498db', linewidth=2, label="CART unpruned original 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Pr_Orig"], marker='d', markersize=7, linestyle='--', color='#2980b9', linewidth=2, label="CART pruned original 3 classes")

# Green Series (All Filters)
ax.plot(df_res["Train_Percentage"], df_res["C4.5_Un_All"], marker='^', markersize=7, linestyle='-.', color='#1e824c', linewidth=2, label="C4.5 unpruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["C4.5_Pr_All"], marker='s', markersize=7, linestyle=':', color='#a2d9ce', linewidth=2, label="C4.5 pruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Un_All"], marker='o', markersize=7, linestyle='-', color='#27ae60', linewidth=2, label="CART unpruned all filters 3 classes")
ax.plot(df_res["Train_Percentage"], df_res["CART_Pr_All"], marker='d', markersize=7, linestyle='--', color='#2ecc71', linewidth=2, label="CART pruned all filters 3 classes")

ax.set_title("Correctly Classified Data in Dependence of the Quantity of Training Data", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Percentage of Training Data (%)", fontsize=11, labelpad=8)
ax.set_ylabel("Correctly Predicted Data (Accuracy)", fontsize=11, labelpad=8)

# EXACT PAPER Y-BOUNDS (%30 - %65)
ax.set_ylim(0.30, 0.65)
ax.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65])

ax.set_xlim(5, 95)
ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

ax.xaxis.set_major_formatter(mtick.PercentFormatter(100.0, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

# LEJAND SAĞ TARAFTA
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="white", framealpha=0.9, edgecolor="#bdc3c7", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(output_dir, "paper_style_sensitivity_curves.png")
plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()

