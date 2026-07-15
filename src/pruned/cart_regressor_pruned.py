import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os # Klasör kontrolü için eklendi

print("=" * 60)
print("STARTING PRUNED CART REGRESSION MODEL...")
print("=" * 60)

# Çıktı klasörünün varlığından emin oluyoruz
os.makedirs("outputs/figures", exist_ok=True)

data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_reg_train"], data["y_reg_test"] # AU Sütunu (Sayısal Hedef)
train_groups = data["train_groups"] # 851 bıçağın grup ID'leri
feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# 42 Özellikli regresyon verisi için parametre aralıkları genişletildi (Pre-pruning)
param_grid = {
    "criterion": ["squared_error", "absolute_error"], # Mutlak hata kriterini de performans için ekledik
    "max_depth": [6, 8, 10, 12],                      # Sığ ağaçlardan kaynaklanan underfitting engellendi
    "min_samples_split": [20, 40, 60],                # 10 satırlık bıçak bloklarına göre esnetildi
    "min_samples_leaf": [10, 15, 25]                  # Regresyon tahminlerini hassaslaştırmak için optimize edildi
}

# Dinamik grup kontrolü ile GroupKFold yapısı
unique_groups = len(np.unique(train_groups))
n_splits = min(5, unique_groups)

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid, 
    cv=GroupKFold(n_splits=n_splits), # Bıçak sızıntısını önleyen grup yapısı korundu
    scoring="neg_mean_squared_error", 
    n_jobs=-1 # İşlemci çekirdekleri tam güç kullanılarak hızlandırıldı
)
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"✓ Best Pruned Parameters: {grid_search.best_params_}")
best_reg = grid_search.best_estimator_
y_pred = best_reg.predict(X_test)

print("\n" + "=" * 60 + "\nPRUNED REGRESSION PERFORMANCE\n" + "=" * 60)
print(f"  MAE: {mean_absolute_error(y_test, y_pred):.4f} μm | R² Score: {r2_score(y_test, y_pred):.4f}")

# Ağacın regresyon kararlarını ve ilk kırılımlarını net görmek için max_depth=4 yapıldı
fig, ax = plt.subplots(figsize=(35, 18), dpi=300)
plot_tree(best_reg, max_depth=4, feature_names=feature_names, filled=True, rounded=True, impurity=True, fontsize=11, ax=ax)
plt.tight_layout()

try:
    plt.savefig("outputs/figures/regression_pruned_tree.png", bbox_inches='tight', dpi=300)
    print("\n✓ Pruned CART regression tree plot saved successfully.")
except OSError as ex:
    print(f"WARNING: Could not save regression tree plot: {ex}")
finally:
    plt.close(fig)