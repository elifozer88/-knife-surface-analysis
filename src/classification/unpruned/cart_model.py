# =============================================================================
# cart_model.py (REVISED TO GENERATE BOTH TREE & MATRIX PLOTS)
# Knife Surface Analysis — CART Decision Tree Based on Gini Impurity
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
print("STARTING GINI IMPURITY-BASED CART MODEL WITH VISUAL OUTPUTS...")
print("=" * 60)

# -----------------------------------------------------------------------------
# DİZİN YAPILANDIRMASI (HEDEF: outputs/figures/unpruned/)
# -----------------------------------------------------------------------------
# Unpruned görsel çıktılarını düzenli olarak bu klasöre yönlendiriyoruz
output_dir = os.path.join("outputs", "figures", "2_unpruned_models")
os.makedirs(output_dir, exist_ok=True)

# Matris için:
matrix_path = os.path.join(output_dir, "Fig3b_cart_unpruned_matrix.png")
# Ağaç için:
tree_path = os.path.join(output_dir, "Fig3a_cart_unpruned_tree.png")

# 1. LOAD THE DATA
data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_clf_train"]
y_test = data["y_clf_test"]
train_groups = data["train_groups"]

feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# 2. GRID SEARCH USING GINI PARAMETERS (UNPRUNED ALIGNMENT)
param_grid = {
    "criterion": ["gini"],             # Gini Impurity odaklı bölünme kriteri
    "ccp_alpha": [0.0],                # Budama kapalı (Unpruned)
    "max_depth": [8, 10, 12, 15, None], # Maksimum derinlik serbest bırakıldı
    "min_samples_split": [2, 5, 10],   # Nesne bölünme eşiği minimuma esnetildi
    "min_samples_leaf": [1, 2, 5]      # Yaprak saflığı optimize edildi
}

# 851 bıçağın 3 sınıfını da katlamalara (folds) eşit dağıtan strateji
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

print("\n-> Computing Gini impurity optimization...")
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"\n✓ Best Gini Parameters: {grid_search.best_params_}")
print(f"✓ Best Mean CV F1-Macro Score: {grid_search.best_score_:.4f}")

# 3. EVALUATE THE MODEL
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Makaledeki Spesifikasyon Limitlerine Göre Sınıflar
class_names = [
    "Class 0 (Ra < 0.11 um)",   # Alt limit altı fire
    "Class 1 (0.11 <= Ra <= 0.45 um)", # Kabul edilebilir kaliteli ürün
    "Class 2 (Ra > 0.45 um)"    # Üst limit üstü fire
]

print("\n" + "=" * 60)
print("GINI MODEL TEST SET PERFORMANCE REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
print("\n--- CONFUSION MATRIX (Terminal View) ---")
print(cm)

# =============================================================================
# 4. PLOT & SAVE CONFUSION MATRIX (AKADEMİK GÖRSEL ÇIKTI)
# =============================================================================
print("\n-> Generating and saving the Confusion Matrix plot...")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6), dpi=300)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2"])
disp.plot(cmap=plt.cm.Greens, values_format="d", ax=ax_cm) # CART için yeşil tonlarında şık bir tema

ax_cm.set_title("CART Gini Model - Confusion Matrix", fontsize=14, pad=15, fontweight='bold')
ax_cm.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax_cm.set_ylabel("True Label", fontsize=12, labelpad=10)
plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/unpruned/ altına kaydediyoruz
matrix_path = os.path.join(output_dir, "cart_gini_confusion_matrix.png")
plt.savefig(matrix_path, bbox_inches='tight', dpi=300)
plt.close(fig_cm)
print(f"✓ Confusion Matrix plot saved to '{matrix_path}'")

# 5. TOP 5 MOST IMPORTANT FEATURES BY GINI IMPURITY REDUCTION
print("\n" + "=" * 60)
print("Top 5 Most Important Features by Gini Impurity Reduction:")
print("=" * 60)
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(5, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]*100:.2f}%")

# =============================================================================
# 6. ACADEMIC-STYLE DECISION TREE VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("Generating the decision tree plot in an academic style...")
print("=" * 60)

fig_tree, ax_tree = plt.subplots(figsize=(35, 18), dpi=300)

plot_tree(
    best_clf, 
    max_depth=3, # İlk kırılımları net görmek için çizim derinliği 3 ile sınırlandı
    feature_names=feature_names, 
    class_names=["Waste (Low)", "Good", "Waste (High)"], 
    filled=True, 
    rounded=True, 
    impurity=True, 
    node_ids=False, 
    proportion=False, 
    fontsize=11, 
    ax=ax_tree
)

plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/unpruned/ altına kaydediyoruz
tree_path = os.path.join(output_dir, "gini_optimized_tree.png")
plt.savefig(tree_path, bbox_inches='tight', dpi=300)
plt.close(fig_tree)
print(f"✓ The academic-style CART tree was saved to '{tree_path}'!")