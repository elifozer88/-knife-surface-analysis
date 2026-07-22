# =============================================================================
# regression_pruned_matrix.py
# Knife Surface Analysis — Binned Evaluation Matrices & Heatmaps for Regressors
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 75)
print("RUNNING PRUNED REGRESSOR EVALUATION & HEATMAP GENERATION...")
print("=" * 75)

# -----------------------------------------------------------------------------
# DIRECTORY CONFIGURATION
# -----------------------------------------------------------------------------
# Heatmap directed to 3_pruned_models, data tables kept in outputs/tables/
fig_output_dir = os.path.join("outputs", "figures", "3_pruned_models")
table_output_dir = os.path.join("outputs", "tables")

os.makedirs(fig_output_dir, exist_ok=True)
os.makedirs(table_output_dir, exist_ok=True)

# 1. LOAD DATA
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. "
        f"Please run 'data_prep.py' first to build split data."
    )

data = np.load(data_path, allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_reg_train"]
y_test = data["y_reg_test"]

# 2. MINI HYPERPARAMETER BENCHMARK FOR PRUNED STRUCTURES
param_grid = [
    {"criterion": "squared_error", "max_depth": 5, "min_samples_split": 30, "min_samples_leaf": 20},
    {"criterion": "squared_error", "max_depth": 4, "min_samples_split": 20, "min_samples_leaf": 15},
    {"criterion": "squared_error", "max_depth": 6, "min_samples_split": 40, "min_samples_leaf": 25},
]

results = []

for idx, params in enumerate(param_grid):
    reg = DecisionTreeRegressor(random_state=42, **params)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append((idx, params, mae, mse, rmse, y_pred))
    print(f"Run {idx+1}: params={params} | MAE={mae:.4f} | RMSE={rmse:.4f}")

# 3. SELECT OPTIMAL PERFORMANCE COHORT BY MAE
best = min(results, key=lambda r: r[2])
best_idx, best_params, best_mae, best_mse, best_rmse, best_pred = best
print(f"\n✓ Selected Optimal Run {best_idx+1} Parameters: {best_params}")

# 4. EXPORT PREDICTIONS METADATA
df = pd.DataFrame({"y_true": y_test, "y_pred": best_pred})
df["residual"] = df["y_true"] - df["y_pred"]
csv_path = os.path.join(table_output_dir, f"pruned_regression_predictions_run{best_idx+1}.csv")
df.to_csv(csv_path, index=False)
print(f"✓ Saved predictions CSV metadata to: '{csv_path}'")

# 5. CREATE BINNED COGNITIVE CROSSTAB MATRIX
n_bins = 10
bins = pd.qcut(df["y_true"], q=n_bins, duplicates='drop')
cols = pd.qcut(df["y_pred"], q=n_bins, duplicates='drop')
ct = pd.crosstab(bins, cols)
ct_path = os.path.join(table_output_dir, f"pruned_regression_matrix_run{best_idx+1}.csv")
ct.to_csv(ct_path)
print(f"✓ Saved binned crosstab matrix CSV to: '{ct_path}'")

# 6. GENERATE PUBLICATIONS-GRADE HEATMAP PLOT
ct_norm = ct.div(ct.sum(axis=1), axis=0)
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(ct_norm, cmap='viridis', cbar=True, annot=False)
plt.title('Binned True vs Predicted (Normalized by True Bin)', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Predicted Roughness Bins ($R_a$)', fontsize=10, labelpad=8)
plt.ylabel('True Roughness Bins ($R_a$)', fontsize=10, labelpad=8)

fig_path = os.path.join(fig_output_dir, f"pruned_regression_matrix_run{best_idx+1}.png")
plt.tight_layout()
plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"✓ Saved normalized regression heatmap successfully to: '{fig_path}'")

# 7. APPEND EXCEL SUMMARY METRICS SHEET
summary_path = "outputs/metrics_summary.csv"
try:
    master = pd.read_csv(summary_path)
except Exception:
    master = pd.DataFrame()

row = {
    "model": "cart_regressor_pruned_matrix",
    "type": "regression",
    "MAE": round(best_mae, 4),
    "RMSE": round(best_rmse, 4),
    "MSE": round(best_mse, 6),
    "R2": "",
    "accuracy": "",
    "macro_f1": "",
    "notes": str(best_params),
    "figure": fig_path
}
master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)
master.to_csv(summary_path, index=False)
print(f"✓ Appended performance metrics successfully to: '{summary_path}'")