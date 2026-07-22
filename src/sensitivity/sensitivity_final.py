# =============================================================================
# sensitivity_final.py (TEK, TUTARLI, ANA PIPELINE ILE HIZALI VERSIYON)
# Knife Surface Analysis — Paper-Style Sensitivity Curve (Fig. 4 Replication)
# Hochschule München / KIM — Erasmus+ Internship Project
#
# BU SCRIPT ESKİ ÜÇ DOSYANIN (sensitivity_runs.py, sensitivity_70_30.py,
# sensitivity_90_10.py) YERİNİ ALIR. Onları artık kullanma / silmeni öneririm.
#
# DÜZELTİLEN 4 HATA:
#   1. Tek split mantığı: data_prep.py'deki ile AYNI random_state=42, sadece
#      test_size (%10-%90) değişiyor. Önceki 3 dosya farklı seed setleri
#      kullandığı için 3 farklı, birbiriyle kıyaslanamaz grafik üretiyordu.
#   2. Model hiperparametreleri artık c45_model.py / cart_model.py / *_pruned.py
#      dosyalarındaki GridSearchCV best_params_ ile BİREBİR AYNI. Önceden
#      sensitivity script'leri kendi uydurduğu (max_depth=12, ccp_alpha=0.001
#      gibi) değerlerle çalışıyordu — ana modellerle alakasızdı.
#   3. "Original features vs All filters" ayrımı KALDIRILDI. Ana pipeline'da
#      (c45_model.py/cart_model.py) böyle bir ayrım yok, 42 feature'ın tamamı
#      tek blok olarak kullanılıyor. Sensitivity de aynısını yapmalı.
#   4. Test grubu artık GERÇEK bıçak ID'leri (train_groups/test_groups) ile
#      birleştiriliyor. Önceki versiyon test örnekleri için sahte, birbirinden
#      bağımsız "grp_i" ID'leri üretiyordu — bu da GroupShuffleSplit'in asıl
#      amacını (aynı bıçağın 10 satırının train/test'e bölünmemesi) resample
#      sırasında geçersiz kılıyordu.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

print("=" * 75)
print("PAPER-STYLE SENSITIVITY CURVE — FINAL, PIPELINE-ALIGNED VERSION")
print("=" * 75)

output_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. VERİYİ YÜKLE (data_prep.py çıktısı — TAM 42 FEATURE, AYRIM YOK)
# -----------------------------------------------------------------------------
data_path = os.path.join("data", "processed", "split_data.npz")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"CRITICAL ERROR: '{data_path}' not found. Önce data_prep.py çalıştır.")

data = np.load(data_path, allow_pickle=True)

# TÜM veriyi tekrar birleştiriyoruz çünkü sensitivity analizi farklı
# train/test oranlarını KENDİSİ yeniden bölecek. Ama artık GERÇEK grup
# (bıçak) ID'leriyle — sahte per-sample ID değil.
X_all = np.vstack([data["X_train"], data["X_test"]])
y_all = np.concatenate([data["y_clf_train"], data["y_clf_test"]])
groups_all = np.concatenate([data["train_groups"], data["test_groups"]]).astype(str)

print(f"✓ Toplam örnek: {X_all.shape[0]} satır, {X_all.shape[1]} feature")
print(f"✓ Toplam benzersiz bıçak (grup): {len(np.unique(groups_all))}")

# -----------------------------------------------------------------------------
# 2. ANA PIPELINE'DAN GELEN best_params_ (GridSearchCV çıktıları — SABİT)
# -----------------------------------------------------------------------------
C45_UNPRUNED_PARAMS = dict(
    criterion="entropy", ccp_alpha=0.0,
    max_depth=15, min_samples_split=10, min_samples_leaf=2,
)
C45_PRUNED_PARAMS = dict(
    criterion="entropy", ccp_alpha=0.001,
    min_samples_split=2, min_samples_leaf=2,
)
CART_UNPRUNED_PARAMS = dict(
    criterion="gini", ccp_alpha=0.0,
    max_depth=8, min_samples_split=5, min_samples_leaf=2,
)
CART_PRUNED_PARAMS = dict(
    criterion="gini", ccp_alpha=0.001,
    min_samples_split=20, min_samples_leaf=1,
)

train_percentages = [0.10, 0.30, 0.50, 0.70, 0.90]
results_db = []

# -----------------------------------------------------------------------------
# 3. TEK SEED MANTIĞI İLE SENSITIVITY DÖNGÜSÜ
# -----------------------------------------------------------------------------
for p in train_percentages:
    test_size = 1.0 - p
    # data_prep.py ile BİREBİR AYNI seed (42) — sadece test_size değişiyor.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups_all))

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    models = {
        "C4.5_Unpruned": DecisionTreeClassifier(class_weight="balanced", random_state=42, **C45_UNPRUNED_PARAMS),
        "C4.5_Pruned":   DecisionTreeClassifier(class_weight="balanced", random_state=42, **C45_PRUNED_PARAMS),
        "CART_Unpruned": DecisionTreeClassifier(class_weight="balanced", random_state=42, **CART_UNPRUNED_PARAMS),
        "CART_Pruned":   DecisionTreeClassifier(class_weight="balanced", random_state=42, **CART_PRUNED_PARAMS),
    }

    row = {"Train_Percentage": p * 100}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        row[name] = accuracy_score(y_test, clf.predict(X_test))
    results_db.append(row)

    print(f"  p={p:.0%} -> " + ", ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "Train_Percentage"))

df_res = pd.DataFrame(results_db)
df_res.to_csv(os.path.join(output_dir, "sensitivity_final_results.csv"), index=False)

# -----------------------------------------------------------------------------
# 4. TEK GRAFİK — PAPER FIG.4 STİLİ
# -----------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

ax.plot(df_res["Train_Percentage"], df_res["C4.5_Unpruned"], marker='^', markersize=8, linestyle='-.', color='#1f3a60', linewidth=2, label="C4.5 unpruned")
ax.plot(df_res["Train_Percentage"], df_res["C4.5_Pruned"],   marker='s', markersize=8, linestyle=':',  color='#5dade2', linewidth=2, label="C4.5 pruned")
ax.plot(df_res["Train_Percentage"], df_res["CART_Unpruned"], marker='o', markersize=8, linestyle='-',  color='#27ae60', linewidth=2, label="CART unpruned")
ax.plot(df_res["Train_Percentage"], df_res["CART_Pruned"],   marker='d', markersize=8, linestyle='--', color='#2ecc71', linewidth=2, label="CART pruned")

ax.set_title("Correctly Classified Data in Dependence of the Quantity of Training Data", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Percentage of Training Data (%)", fontsize=11, labelpad=8)
ax.set_ylabel("Correctly Predicted Data (Accuracy)", fontsize=11, labelpad=8)

ax.set_xlim(5, 95)
ax.set_xticks([10, 30, 50, 70, 90])
ax.xaxis.set_major_formatter(mtick.PercentFormatter(100.0, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, facecolor="white", framealpha=0.9, edgecolor="#bdc3c7", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(output_dir, "paper_style_sensitivity_curve_FINAL.png")
plt.savefig(fig_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n✓ Nihai grafik kaydedildi: '{fig_path}'")
print(f"✓ Sayısal sonuçlar: '{os.path.join(output_dir, 'sensitivity_final_results.csv')}'")