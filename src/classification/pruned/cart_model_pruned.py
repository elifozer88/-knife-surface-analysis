# =============================================================================
# cart_model_pruned.py (REVISED TO MATCH EXPERIMENTAL POST-PRUNING METHODOLOGY)
# Knife Surface Analysis — Academic CART Pruned Simulator using Gini & CCP Pruning
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("STARTING ACADEMIC GINI CART PRUNED MODEL WITH VISUAL OUTPUTS...")
print("=" * 60)

# -----------------------------------------------------------------------------
# DİZİN YAPILANDIRMASI (HEDEF: outputs/figures/pruned/)
# -----------------------------------------------------------------------------
# Çıktıların düzenli olarak pruned klasörüne gitmesi için hedef yolu tanımlıyoruz
output_dir = os.path.join("outputs", "figures", "3_pruned_models")
os.makedirs(output_dir, exist_ok=True)

# Matris için:
matrix_path = os.path.join(output_dir, "Fig6b_cart_pruned_matrix.png")
# Ağaç için:
tree_path = os.path.join(output_dir, "Fig6a_cart_pruned_tree.png")

# 1. VERİYİ YÜKLE
data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_clf_train"], data["y_clf_test"] 
train_groups = data["train_groups"] 
feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# 2. POST-PRUNING DESTEKLİ PARAMETRE ALANI
param_grid = {
    "criterion": ["gini"],
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.015, 0.02], # Gerçek budama (pruning) etkisi
    "min_samples_split": [2, 5, 10, 20],                # Nesne bölünme limitleri (minNumObj=2 dahil)
    "min_samples_leaf": [1, 2, 5, 10]                   # Yaprak düğüm hassasiyet sınırları
}

# 3 Sınıfın katlamalara dengeli dağılması ve grup sızıntısının önlenmesi
unique_groups = len(np.unique(train_groups))
n_splits = min(5, unique_groups)
cv = StratifiedGroupKFold(n_splits=n_splits)

base_clf = DecisionTreeClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid, 
    cv=cv, 
    scoring="f1_macro", 
    n_jobs=-1,
    verbose=1
)

print("\n-> Computing Gini CART pruned optimization...")
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"\n✓ Best Gini CART Pruned Parameters: {grid_search.best_params_}")
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Makaledeki Spesifikasyon Sınırlarına Göre Sınıflar
class_names = [
    "Class 0 (Ra < 0.11 um - Waste)", 
    "Class 1 (0.11 <= Ra <= 0.45 um)", 
    "Class 2 (Ra > 0.45 um - Waste)"
]

print("\n" + "=" * 60 + "\nREVISED GINI CART PRUNED PERFORMANCE REPORT\n" + "=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
print("\n--- CONFUSION MATRIX (Terminal View) ---")
print(cm)

# =============================================================================
# 3. PLOT & SAVE CONFUSION MATRIX (AKADEMİK GÖRSEL ÇIKTI)
# =============================================================================
print("\n-> Generating and saving the Pruned Gini Confusion Matrix plot...")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6), dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"])
disp.plot(cmap=plt.cm.Greens, values_format="d", ax=ax_cm) # CART / Gini için şık yeşil tonları

ax_cm.set_title("Gini CART Pruned Model - Confusion Matrix", fontsize=14, pad=15, fontweight='bold')
ax_cm.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax_cm.set_ylabel("True Label", fontsize=12, labelpad=10)
plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/pruned/ altına kaydediyoruz
matrix_path = os.path.join(output_dir, "gini_pruned_confusion_matrix.png")
plt.savefig(matrix_path, bbox_inches='tight', dpi=300)
plt.close(fig_cm)
print(f"✓ Confusion Matrix plot saved successfully to '{matrix_path}'")

# 4. EN ÖNEMLİ 5 ÖZELLİK (Gini Azalımı ile)
print("\n" + "=" * 60)
print("Top 5 Most Important Features by Gini Impurity (Pruned):")
print("=" * 60)
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(min(5, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]*100:.2f}%")

# =============================================================================
# 5. AKADEMİK STİLDE KARAR AĞACI ÇİZİMİ
# =============================================================================
print("\n" + "=" * 60)
print("Generating the academic-style CART pruned tree plot...")
print("=" * 60)

fig_tree, ax_tree = plt.subplots(figsize=(30, 15), dpi=300)
plot_tree(
    best_clf, 
    max_depth=3, # Ağacın karmaşıklığını önleyip okunabilir kılmak adına görsel derinliği 3 yaptık
    feature_names=feature_names, 
    class_names=["Waste (Low)", "Good", "Waste (High)"], 
    filled=True, 
    rounded=True, 
    impurity=True, # Her düğümde Gini değerini gösterir (G = ...)
    fontsize=10, 
    ax=ax_tree
)
plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/pruned/ altına kaydediyoruz
tree_path = os.path.join(output_dir, "gini_pruned_tree.png")
try:
    plt.savefig(tree_path, bbox_inches='tight', dpi=300)
    print(f"\n✓ Academic Gini CART Decision Tree plot saved successfully to '{tree_path}'.")
except OSError as ex:
    print(f"WARNING: Could not save tree plot: {ex}")
finally:
    plt.close(fig_tree)