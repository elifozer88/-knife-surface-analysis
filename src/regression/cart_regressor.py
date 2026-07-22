# =============================================================================
# cart_regressor.py (REVISED TO FOCUS EXCLUSIVELY ON MSE / SQUARED ERROR)
# Knife Surface Analysis — CART Regression Model (Ra Prediction)
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("STARTING MSE-FOCUSED CART REGRESSION MODEL...")
print("=" * 60)

# -----------------------------------------------------------------------------
# DIRECTORY CONFIGURATION (TARGET: outputs/figures/2_unpruned_models/)
# -----------------------------------------------------------------------------
# Directing unpruned regression outputs to our standard unpruned repository
output_dir = os.path.join("outputs", "figures", "2_unpruned_models")
os.makedirs(output_dir, exist_ok=True)

# 1. LOAD THE DATA
data_path = "data/processed/split_data.npz"
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"CRITICAL ERROR: '{data_path}' not found. "
        f"Please run 'data_prep.py' first to build dataset partitions."
    )

data = np.load(data_path, allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_reg_train"]  # Target: Continuous roughness values (Ra)
y_test = data["y_reg_test"]
train_groups = data["train_groups"]

feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

print(f"✓ Regression data loaded successfully.")
print(f"  Target range (Ra): min={y_train.min():.4f} μm, max={y_train.max():.4f} μm")

# 2. HYPERPARAMETER OPTIMIZATION (EXCLUSIVELY SQUARED ERROR / MSE)
param_grid = {
    "criterion": ["squared_error"],             # Direct focus on MSE optimization
    "ccp_alpha": [0.0],                         # Pruning disabled (Unpruned)
    "max_depth": [4, 6, 8, 10, 12, None],        # Depth limit ranges
    "min_samples_split": [2, 5, 10, 20],        # Split sample ranges
    "min_samples_leaf": [1, 2, 5, 10, 15]       # Leaf configuration ranges
}

# Preserve knife specimen group isolation to prevent leakages
unique_groups = len(np.unique(train_groups))
n_splits = min(5, unique_groups)
cv = GroupKFold(n_splits=n_splits)

base_regressor = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=base_regressor,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_mean_squared_error",  # Force optimization based on Mean Squared Error
    n_jobs=-1,
    verbose=1
)

print("\n-> Computing MSE-based regression optimization...")
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"\n✓ Best MSE Regression Parameters: {grid_search.best_params_}")

# 3. TEST SET PERFORMANCE EVALUATION
best_reg = grid_search.best_estimator_
y_pred = best_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("CART REGRESSION MODEL TEST SET PERFORMANCE")
print("=" * 60)
print(f"  Mean Absolute Error (MAE)      : {mae:.4f} μm")
print(f"  Mean Squared Error (MSE)       : {mse:.6f}")
print(f"  Root Mean Squared Error (RMSE) : {rmse:.4f} μm")
print(f"  R² Score (Açıklayıcılık Oranı) : {r2:.4f}")

# 4. TOP 5 MOST IMPORTANT FEATURES BY MSE REDUCTION
print("\n" + "=" * 60)
print("Top 5 Most Important Features by MSE (Variance) Reduction:")
print("=" * 60)
importances = best_reg.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(5, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]*100:.2f}%")

# =============================================================================
# 5. ACADEMIC-STYLE REGRESSION TREE VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("Generating the MSE regression tree plot in an academic style...")
print("=" * 60)

fig, ax = plt.subplots(figsize=(35, 18), dpi=300)

plot_tree(
    best_reg,
    max_depth=3,  # Restricted visualization depth for paper legibility
    feature_names=feature_names,
    filled=True,
    rounded=True,
    impurity=True,  # Displays local node MSE variance explicitly
    fontsize=11,
    ax=ax
)

plt.tight_layout()

# SAVED PATH: Directed to 2_unpruned_models
tree_path = os.path.join(output_dir, "regression_cart_tree.png")
plt.savefig(tree_path, bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"✓ MSE-based regression tree plot saved successfully to '{tree_path}'.")