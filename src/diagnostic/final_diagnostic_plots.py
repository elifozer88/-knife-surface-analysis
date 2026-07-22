# =============================================================================
# final_diagnostic_plots.py
# Knife Surface Analysis — Final Matrix Geometry & Residual Histogram Diagnostics
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import os

# Securely import seaborn for publication-quality statistical plots
try:
    import seaborn as sns
except ImportError:
    raise ImportError("Please install seaborn via: pip install seaborn")

print("=" * 70)
print("GENERATING FINAL MATRIX GEOMETRY & HISTOGRAM DIAGNOSTICS...")
print("=" * 70)

# -----------------------------------------------------------------------------
# DIRECTORY CONFIGURATION (TARGET: outputs/figures/4_sensitivity_analysis/)
# -----------------------------------------------------------------------------
# Saving the vertical alignment diagnostics in our dedicated sensitivity folder
output_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. LOAD THE DATASETS
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. "
        f"Please run 'data_prep.py' first to generate required splits."
    )

data = np.load(data_path, allow_pickle=True)
X_all = np.vstack([data["X_train"], data["X_test"]])
y_clf_all = np.concatenate([data["y_clf_train"], data["y_clf_test"]])
y_reg_all = np.concatenate([data["y_reg_train"], data["y_reg_test"]])
groups_all = np.arange(len(X_all))

scenarios = {"70_30": 0.30, "80_20": 0.20, "90_10": 0.10}

# Configure a vertical 3x1 canvas for textbook presentation
fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300)
axes = axes.flatten()

# =============================================================================
# 2. ITERATIVE MODEL TRAINING & HISTOGRAM PLOTTING
# =============================================================================
for idx, (label, test_size) in enumerate(scenarios.items()):
    # Apply Group-based splitting to eliminate data leakage across profiles
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_reg_all, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_reg_train, y_reg_test = y_reg_all[train_idx], y_reg_all[test_idx]
    
    # Train the optimized CART Regressor using squared error minimization
    reg = DecisionTreeRegressor(
        criterion="squared_error", 
        max_depth=6, 
        min_samples_leaf=15, 
        random_state=42
    )
    reg.fit(X_train, y_reg_train)
    y_pred_reg = reg.predict(X_test)
    
    # Plot raw frequency distributions for comparative diagnostics
    sns.histplot(y_reg_test, color="darkblue", alpha=0.5, label="Actual Ra", ax=axes[idx], kde=True, bins=30)
    sns.histplot(y_pred_reg, color="crimson", alpha=0.5, label="Predicted Ra", ax=axes[idx], kde=True, bins=30)
    
    # Render academic titles and metadata labels
    axes[idx].set_title(f"Ra Distribution Alignment — Split Scenario: {label.replace('_', '/')}", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Surface Roughness (Ra in μm)", fontsize=10)
    axes[idx].set_ylabel("Frequency (Sample Count)", fontsize=10)
    axes[idx].legend()
    axes[idx].grid(True, linestyle="--", alpha=0.5)

# 3. EXPORT FINAL DIAGNOSTIC GRAPHIC TO SERIALIZED WORKSPACE
plt.tight_layout()
histogram_path = os.path.join(output_dir, "Fig10_vertical_ra_alignment_diagnostics.png")
plt.savefig(histogram_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Vertical Ra alignment diagnostics saved successfully to: '{histogram_path}'")