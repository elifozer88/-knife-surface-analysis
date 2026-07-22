# =============================================================================
# c45_model.py (REVISED TO GENERATE BOTH TREE & MATRIX PLOTS)
# Knife Surface Analysis — C4.5-Based Model (Entropy / Information Gain)
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
print("STARTING C4.5 (ENTROPY-BASED UNPRUNED) MODEL WITH VISUAL OUTPUTS...")
print("=" * 60)

# -----------------------------------------------------------------------------
# DİZİN YAPILANDIRMASI (HEDEF: outputs/figures/unpruned/)
# -----------------------------------------------------------------------------
# Çıktıların düzenli olarak unpruned klasörüne gitmesi için hedef yolu tanımlıyoruz
output_dir = os.path.join("outputs", "figures", "2_unpruned_models")
os.makedirs(output_dir, exist_ok=True)

# Matris için:
matrix_path = os.path.join(output_dir, "Fig2b_c45_unpruned_matrix.png")
# Ağaç için:
tree_path = os.path.join(output_dir, "Fig2a_c45_unpruned_tree.png")

# 1. LOAD THE DATA
data = np.load("data/processed/split_data.npz", allow_pickle=True)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_clf_train"]
y_test = data["y_clf_test"]
train_groups = data["train_groups"]

feature_names = pd.read_csv("data/processed/feature_names.csv").iloc[:, 0].tolist()

# 2. HYPERPARAMETER OPTIMIZATION (UNPRUNED J48 SIMULATION)
param_grid = {
    "criterion": ["entropy"],          # C4.5 / Bilgi Kazancı temeli
    "ccp_alpha": [0.0],                # Budama kapalı (Unpruned)
    "max_depth": [8, 10, 12, 15, None], # Derin dallanmaya izin veriliyor
    "min_samples_split": [2, 5, 10],   # J48 varsayılan nesne sınırı (minNumObj=2) dahil
    "min_samples_leaf": [1, 2, 5]      # Saflık sınırına kadar gitme serbestisi
}

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

print("\n-> Computing C4.5 unpruned entropy optimization...")
grid_search.fit(X_train, y_train, groups=train_groups)

print(f"\n✓ Best C4.5 Unpruned Parameters: {grid_search.best_params_}")
print(f"✓ Best Mean CV F1-Macro Score: {grid_search.best_score_:.4f}")

# 3. EVALUATE THE TEST SET
best_c45 = grid_search.best_estimator_
y_pred = best_c45.predict(X_test)

# Makale Limitleri: Class 0 (Ra < 0.11), Class 1 (0.11 <= Ra <= 0.45), Class 2 (Ra > 0.45)
class_names = [
    "Class 0 (Ra < 0.11 um)", 
    "Class 1 (0.11 <= Ra <= 0.45 um)", 
    "Class 2 (Ra > 0.45 um)"
]

print("\n" + "=" * 60)
print("C4.5 UNPRUNED MODEL TEST SET PERFORMANCE REPORT")
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
disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax_cm)

ax_cm.set_title("C4.5 Unpruned Model - Confusion Matrix", fontsize=14, pad=15, fontweight='bold')
ax_cm.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax_cm.set_ylabel("True Label", fontsize=12, labelpad=10)
plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/unpruned/ altına kaydediyoruz
matrix_path = os.path.join(output_dir, "c45_unpruned_confusion_matrix.png")
plt.savefig(matrix_path, bbox_inches='tight', dpi=300)
plt.close(fig_cm)
print(f"✓ Confusion Matrix plot saved to '{matrix_path}'")

# 5. TOP 5 MOST IMPORTANT FEATURES BY INFORMATION GAIN
print("\n" + "=" * 60)
print("Top 5 Most Important Features by Information Gain (Unpruned):")
print("=" * 60)
importances = best_c45.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(5, len(feature_names))):
    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]*100:.2f}%")

# =============================================================================
# 6. ACADEMIC-STYLE C4.5 TREE VISUALIZATION
# =============================================================================
print("\n" + "=" * 60)
print("Generating the C4.5 unpruned tree plot in an academic style...")
print("=" * 60)

fig_tree, ax_tree = plt.subplots(figsize=(35, 18), dpi=300)

plot_tree(
    best_c45, 
    max_depth=3,                    # Görsel anlaşılabilirlik için çizim derinliği
    feature_names=feature_names, 
    class_names=["Waste (Low)", "Good", "Waste (High)"], 
    filled=True, 
    rounded=True, 
    impurity=True,                  # Kutularda Entropy (E = ...) gösterimi
    node_ids=False, 
    proportion=False, 
    fontsize=11, 
    ax=ax_tree
)

plt.tight_layout()

# GÜNCELLENEN KISIM: Görseli outputs/figures/unpruned/ altına kaydediyoruz
tree_path = os.path.join(output_dir, "c45_unpruned_tree.png")
plt.savefig(tree_path, bbox_inches='tight', dpi=300)
plt.close(fig_tree)
print(f"✓ The academic-style C4.5 unpruned tree was saved to '{tree_path}'!")