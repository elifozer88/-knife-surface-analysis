# =============================================================================
# draw_textbook_comparison.py
# Knife Surface Analysis — Textbook-Style 1x3 Sensitivity Histogram Plot
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Securely import seaborn for publication-quality statistical plots
try:
    import seaborn as sns
except ImportError:
    raise ImportError("Please install seaborn via: pip install seaborn")

print("=" * 70)
print("GENERATING TEXTBOOK-STYLE 1x3 COMPARISON HISTOGRAMS...")
print("=" * 70)

# -----------------------------------------------------------------------------
# DIRECTORY CONFIGURATION (TARGET: outputs/figures/4_sensitivity_analysis/)
# -----------------------------------------------------------------------------
# Organizing the comparison chart under the sensitivity analysis folder
output_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. LOAD THE DATASETS AND ATTRIBUTES
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. "
        f"Please run the data_prep.py script before running this comparison analysis."
    )

data = np.load(data_path, allow_pickle=True)
X_all = np.vstack([data["X_train"], data["X_test"]])
y_reg_all = np.concatenate([data["y_reg_train"], data["y_reg_test"]])
groups_all = np.arange(len(X_all))

# Define systematic holdout testing scenarios
scenarios = {
    "70/30 Split": 0.30,
    "80/20 Split": 0.20,
    "90/10 Split": 0.10
}

# Apply academic styling standards
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300, sharex=True)

# Cohesive academic color scheme: Slate navy for actual data, soft coral for predictions
color_actual = "#2c3e50"
color_pred = "#e74c3c"

# =============================================================================
# 2. ITERATIVE EVALUATION AND HISTOGRAM PLOTTING
# =============================================================================
for idx, (label, test_size) in enumerate(scenarios.items()):
    ax = axes[idx]
    
    # Execute Group-based partition sequence to preserve specimen isolation
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_reg_all, groups=groups_all))
    
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_reg_train, y_reg_test = y_reg_all[train_idx], y_reg_all[test_idx]
    
    # Initialize the optimized CART Regressor utilizing calibrated parameters
    reg = DecisionTreeRegressor(
        criterion="squared_error", 
        max_depth=6, 
        min_samples_leaf=15, 
        random_state=42
    )
    reg.fit(X_train, y_reg_train)
    y_pred = reg.predict(X_test)
    
    # Compute continuous statistical metrics
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    
    # Plot probability density distributions (using density scaling for comparative normalization)
    sns.histplot(y_reg_test, color=color_actual, alpha=0.35, label="Actual Ra", 
                 kde=True, stat="density", bins=25, ax=ax, element="step")
    sns.histplot(y_pred, color=color_pred, alpha=0.45, label="Predicted Ra", 
                 kde=True, stat="density", bins=25, ax=ax, element="step")
    
    # Superimpose academic specification boundaries (0.11 and 0.45 μm limits)
    ax.axvline(x=0.11, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Lower Limit (0.11 μm)")
    ax.axvline(x=0.45, color="#7f8c8d", linestyle="-.", linewidth=1.2, label="Upper Limit (0.45 μm)")
    
    # Configure formal subplot labels and parameters
    ax.set_title(label, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Surface Roughness ($R_a$ in $\mu$m)", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Probability Density", fontsize=11)
    else:
        ax.set_ylabel("") # Hide y-labels on adjacent plots to reduce visual clutter
        
    # Render academic metrics annotation box on each subplot panel
    textstr = f"$R^2$: {r2:.4f}\nMAE: {mae:.4f} $\mu$m"
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#bdc3c7')
    ax.text(0.55, 0.92, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold', color="#2c3e50")
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(0.05, 0.48) # Focus limits strictly on the specification domain

# Construct a single unified legend block placed at the bottom of the canvas
handles, labels = axes[0].get_legend_handles_labels()
# Filter out duplicate handles to guarantee clean labeling
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.02),
           shadow=True, ncol=4, fontsize=11, frameon=True)

plt.tight_layout()
# Allocate proportional canvas buffer space at the bottom for the unified legend
plt.subplots_adjust(bottom=0.15)

# Save high-resolution, textbook-quality comparison plot under the correct path
plot_path = os.path.join(output_dir, "Fig9_textbook_sensitivity_comparison.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Textbook comparison plot saved successfully to: '{plot_path}'")