# Small utility to train a few pruned CART regressors and save prediction matrix
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# 1. Load data
data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_reg_train"]
y_test = data["y_reg_test"]

# 2. Small hyperparameter list to 'play with' (keeps structure minimal)
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

# 3. Select best by MAE
best = min(results, key=lambda r: r[2])
best_idx, best_params, best_mae, best_mse, best_rmse, best_pred = best
print("\nBest params:", best_params)

# 4. Save prediction vs true table
df = pd.DataFrame({"y_true": y_test, "y_pred": best_pred})
df["residual"] = df["y_true"] - df["y_pred"]
csv_path = f"outputs/tables/pruned_regression_predictions_run{best_idx+1}.csv"
df.to_csv(csv_path, index=False)
print(f"Saved predictions CSV to {csv_path}")

# 5. Create binned prediction-vs-true matrix (10 quantile bins)
n_bins = 10
bins = pd.qcut(df["y_true"], q=n_bins, duplicates='drop')
cols = pd.qcut(df["y_pred"], q=n_bins, duplicates='drop')
ct = pd.crosstab(bins, cols)
ct_path = f"outputs/tables/pruned_regression_matrix_run{best_idx+1}.csv"
ct.to_csv(ct_path)
print(f"Saved binned matrix CSV to {ct_path}")

# 6. Plot heatmap of the crosstab (normalized)
ct_norm = ct.div(ct.sum(axis=1), axis=0)
plt.figure(figsize=(8, 6), dpi=200)
sns.heatmap(ct_norm, cmap='viridis', cbar=True)
plt.title('Binned True vs Predicted (normalized by true bin)')
plt.xlabel('Predicted bins')
plt.ylabel('True bins')
fig_path = f"outputs/figures/pruned_regression_matrix_run{best_idx+1}.png"
plt.tight_layout()
plt.savefig(fig_path)
plt.close()
print(f"Saved heatmap to {fig_path}")

# 7. Also save summary metrics to outputs/metrics_summary.csv (append)
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
print(f"Appended summary to {summary_path}")
