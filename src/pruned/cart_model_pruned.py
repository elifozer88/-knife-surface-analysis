import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold # Performans ve 3 sınıf dengesi için güncellendi
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os # Klasör kontrolü için eklendi

sys.path.append("src")
from confusion_matrix_utils import print_terminal_confusion_matrix, save_confusion_matrix_image

print("=" * 60)
print("STARTING PRUNED GINI CART MODEL...")
print("=" * 60)

# Çıktı klasörünün varlığından emin oluyoruz
os.makedirs("outputs/figures", exist_ok=True)

# Load the dataset from the parent directory
data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_clf_train"], data["y_clf_test"] # AT Sütunu (3 Sınıf)
train_groups = data["train_groups"] # 851 bıçağın grup ID'leri
feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# Pruning parameter grid (42 özellik ve 3 sınıf için aralıklar genişletildi)
param_grid = {
    "criterion": ["gini"],
    "max_depth": [6, 8, 10, 12],          # Derinlik artırılarak underfitting engellendi
    "min_samples_split": [20, 40, 60],    # 10'ar satırlık bıçak bloklarına göre esnetildi
    "min_samples_leaf": [10, 15, 25]      # İnce detayları yakalamak için optimize edildi
}

# 3 sınıfın her katlamaya eşit dağılması için StratifiedGroupKFold
unique_groups = len(np.unique(train_groups))
n_splits = min(5, unique_groups)

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(class_weight="balanced", random_state=42),
    param_grid=param_grid, 
    cv=StratifiedGroupKFold(n_splits=n_splits), # Hem grup yapısını hem sınıf dengesini korur
    scoring="f1_macro", 
    n_jobs=-1 # İşlemciyi tam performans çalıştırmak için -1 yapıldı
)
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"✓ Best Pruned Parameters: {grid_search.best_params_}")
best_clf = grid_search.best_estimator_

# Performance report
y_pred = best_clf.predict(X_test)
class_names = ["Class 0", "Class 1", "Class 2"] # AT Sütunundaki 3 sınıfımız
print("\n" + "=" * 60 + "\nPRUNED TEST SET PERFORMANCE REPORT\n" + "=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))
cm = confusion_matrix(y_test, y_pred)
print("\n--- CONFUSION MATRIX ---\n", cm)
print_terminal_confusion_matrix(cm, class_names)
save_confusion_matrix_image(cm, class_names, "outputs/figures/gini_pruned_confusion_matrix.png")
print("\n✓ Colored confusion matrix handling completed.")

# Save the visualization under outputs/figures/
fig, ax = plt.subplots(figsize=(35, 18), dpi=300)
# Ağacın en tepesindeki en can alıcı kararları net görmek için max_depth=4 yapıldı
plot_tree(best_clf, max_depth=4, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, impurity=True, fontsize=11, ax=ax)
plt.tight_layout()
try:
    plt.savefig("outputs/figures/gini_pruned_tree.png", bbox_inches='tight', dpi=300)
    print("\n✓ Pruned CART tree plot saved.")
except OSError as ex:
    print(f"WARNING: Could not save CART tree plot: {ex}")
finally:
    plt.close(fig)