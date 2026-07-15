# =============================================================================
# data_prep.py
# Bıçak Yüzey Analizi — Veri Hazırlık Scripti
# Hochschule München / KIM — Erasmus+ Staj Projesi
# =============================================================================

# -----------------------------------------------------------------------------
# KÜTÜPHANELER
# -----------------------------------------------------------------------------
# pandas: Excel okuma ve veri manipülasyonu için
# numpy: Sayısal işlemler için
# sklearn: Train/test split ve cross-validation için
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# =============================================================================
# ADIM 1: VERİYİ YÜKLE
# =============================================================================
# Neden openpyxl? pandas, .xlsx dosyalarını okumak için arka planda
# openpyxl kütüphanesini kullanır. Bunu pip ile kurman gerekiyor:
# pip install openpyxl
# =============================================================================

print("=" * 60)
print("ADIM 1: Veri Yükleniyor...")
print("=" * 60)

FILE_PATH = "data/raw/Kochmesser_ohne_prozessdaten.xlsx"

df = pd.read_excel(FILE_PATH, engine="openpyxl")

print(f"✓ Dosya yüklendi.")
print(f"  Toplam satır : {df.shape[0]}")
print(f"  Toplam sütun : {df.shape[1]}")


# =============================================================================
# ADIM 2: SÜTUNLARI TANIMLA
# =============================================================================
# Dokümanımızda 3 grup sütun var:
#
#   KİMLİK SÜTUNLARI:
#     - Spalte1 : Satır numarası (modelde kullanılmaz)
#     - Name    : Bıçak ID'si (knife_id olarak kullanacağız)
#     - Linie   : 0-9 arası ölçüm çizgisi numarası
#
#   FEATURE SÜTUNLARI (42 adet, D'den AT'ye kadar = index 3-44):
#     - İlk 12: Derinlik tahminleri (Ra_ganz, Rq_ganz, Rz_ganz ...)
#     - Son 30: Genişlik & miktar özellikleri (Original_, Sobel_, DFT_...)
#
#   HEDEF SÜTUNLAR:
#     - AT sütunu = Ra3 → Sınıflandırma hedefi (0, 1, 2)
#     - AU sütunu = Ra  → Regresyon hedefi (sürekli μm değerleri)
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 2: Sütunlar Tanımlanıyor...")
print("=" * 60)

# Bıçak kimliği — GroupShuffleSplit için kullanacağız
# Name sütununu direkt kullanıyoruz, çünkü zaten her bıçak için benzersiz
KNIFE_ID_COL = "Name"

# 42 feature sütunu: Excel'de D'den AT'ye = pandas'ta index 3'ten 44'e
# df.columns[3:45] → 3 dahil, 45 hariç → tam 42 sütun
FEATURE_COLS = df.columns[3:45].tolist()

# Hedef sütunlar
TARGET_CLF = "Ra3"   # Classification (AT sütunu) — 3 sınıf: 0, 1, 2
TARGET_REG = "Ra"    # Regression (AU sütunu) — sürekli Ra değeri (μm)

print(f"✓ Knife ID sütunu  : {KNIFE_ID_COL}")
print(f"✓ Feature sayısı   : {len(FEATURE_COLS)}")
print(f"✓ Sınıflandırma    : {TARGET_CLF}")
print(f"✓ Regresyon        : {TARGET_REG}")
print(f"\n  Feature listesi:")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    [{i:2d}] {col}")


# =============================================================================
# ADIM 3: TEMEL VERİ KALİTE KONTROLÜ
# =============================================================================
# Modele girmeden önce verimizin sağlıklı olup olmadığını kontrol ediyoruz.
# Eksik veri varsa model hata verir veya yanlış öğrenir.
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 3: Veri Kalite Kontrolü")
print("=" * 60)

# Feature sütunlarında eksik veri var mı?
missing_features = df[FEATURE_COLS].isnull().sum()
missing_features = missing_features[missing_features > 0]

if len(missing_features) == 0:
    print("✓ 42 feature sütununun tamamında eksik veri YOK.")
else:
    print("⚠ Eksik veri bulundu:")
    print(missing_features)

# Hedef sütunlarda eksik veri var mı?
missing_targets = df[[TARGET_CLF, TARGET_REG]].isnull().sum()
print(f"\n✓ Hedef sütun eksik veri kontrolü:")
print(f"  {TARGET_CLF} (sınıf) : {missing_targets[TARGET_CLF]} eksik")
print(f"  {TARGET_REG} (sürekli): {missing_targets[TARGET_REG]} eksik")

# Bıçak başına satır sayısı — her bıçakta tam 10 olmalı
knife_counts = df[KNIFE_ID_COL].value_counts()
print(f"\n✓ Bıçak başına ölçüm çizgisi:")
print(f"  Benzersiz bıçak sayısı : {knife_counts.nunique() if hasattr(knife_counts, 'nunique') else len(knife_counts)}")
print(f"  Her bıçakta satır      : min={knife_counts.min()}, max={knife_counts.max()}, ortalama={knife_counts.mean():.1f}")

# Sınıf dağılımı — dengesiz mi?
print(f"\n✓ {TARGET_CLF} sınıf dağılımı:")
class_dist = df[TARGET_CLF].value_counts().sort_index()
for cls, count in class_dist.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  Sınıf {cls}: {count:5d} satır ({pct:.1f}%) {bar}")

# UYARI: Sınıf 1 (4480 satır) diğerlerine göre çok daha fazla.
# Bu dengesizlik (class imbalance) modeli Sınıf 1'e doğru önyargılı kılar.
# CART'ta class_weight='balanced' parametresiyle bunu dengeleyeceğiz.
print("\n  ⚠ NOT: Sınıf 1 orantısız fazla (class imbalance).")
print("    → CART modelinde class_weight='balanced' kullanılacak.")

# Ra regresyon hedefinin özet istatistikleri
print(f"\n✓ {TARGET_REG} (regresyon) özet istatistikleri:")
print(df[TARGET_REG].describe().to_string())


# =============================================================================
# ADIM 4: X ve y MATRİSLERİNİ OLUŞTUR
# =============================================================================
# Makine öğrenmesinde standart gösterim:
#   X = girdi matrisi (features) — model bunları görür
#   y = çıktı vektörü (hedef) — model bunu tahmin etmeye çalışır
#   groups = bıçak ID'leri — train/test split için kullanılır
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 4: X ve y Matrisleri Oluşturuluyor...")
print("=" * 60)

X = df[FEATURE_COLS].values          # shape: (8510, 42) — numpy array
y_clf = df[TARGET_CLF].values        # shape: (8510,)  — sınıf etiketleri
y_reg = df[TARGET_REG].values        # shape: (8510,)  — sürekli Ra değerleri
groups = df[KNIFE_ID_COL].values     # shape: (8510,)  — knife ID'leri

print(f"✓ X shape     : {X.shape}  (satır x feature)")
print(f"✓ y_clf shape : {y_clf.shape}  (sınıf etiketleri)")
print(f"✓ y_reg shape : {y_reg.shape}  (Ra değerleri)")
print(f"✓ groups shape: {groups.shape}  (knife ID'leri)")


# =============================================================================
# ADIM 5: GRUP BAZLI TRAIN/TEST SPLIT
# =============================================================================
# NEDEN GRUP BAZLI? — Bu projenin en kritik metodolojik kararı.
#
# 851 bıçak × 10 çizgi = 8510 satır.
# Eğer normal random split kullansaydık:
#   → Aynı bıçağın bazı çizgileri TRAIN'de, bazıları TEST'te olurdu.
#   → Model aynı bıçağı hem öğrenme hem sınav aşamasında görürdü.
#   → Bu DATA LEAKAGE (veri sızıntısı) dır — model gerçekte olduğundan
#     çok daha iyi görünür ama gerçek bıçaklara genelleşemez.
#
# GroupShuffleSplit ne yapar?
#   → Bıçakları (groups) bütün olarak TRAIN veya TEST'e atar.
#   → Bir bıçağın 10 çizgisinin TAMAMI ya train'de ya test'te olur.
#   → Test seti modelin hiç görmediği bıçaklardan oluşur.
#   → Bu gerçek hayat senaryosunu simüle eder: yeni bıçak geldi, tahmin et.
#
# test_size=0.2 → %80 train (%80 × 851 ≈ 681 bıçak),
#                 %20 test  (%20 × 851 ≈ 170 bıçak)
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 5: Grup Bazlı Train/Test Split")
print("=" * 60)
print("  (GroupShuffleSplit — knife_id bazında, data leakage önleniyor)")

gss = GroupShuffleSplit(
    n_splits=1,       # 1 kez böl (tek bir train/test ihtiyacımız var)
    test_size=0.2,    # %20 test
    random_state=42   # Tekrar üretilebilirlik için sabit seed
)

# next() ile tek seferlik split alıyoruz
train_idx, test_idx = next(gss.split(X, y_clf, groups=groups))

# Split sonuçlarını uygula
X_train, X_test     = X[train_idx],     X[test_idx]
y_clf_train, y_clf_test = y_clf[train_idx], y_clf[test_idx]
y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

# Kaç benzersiz bıçak train/test'e düştü?
train_knives = len(set(groups[train_idx]))
test_knives  = len(set(groups[test_idx]))

print(f"\n✓ Train seti:")
print(f"  Satır sayısı : {X_train.shape[0]}  ({train_knives} benzersiz bıçak × 10 çizgi)")
print(f"✓ Test seti:")
print(f"  Satır sayısı : {X_test.shape[0]}  ({test_knives} benzersiz bıçak × 10 çizgi)")

# Önemli kontrol: train ve test'teki bıçaklar kesişiyor mu?
# (Hiç kesişmemesi gerekiyor — veri sızıntısı kontrolü)
train_knife_set = set(groups[train_idx])
test_knife_set  = set(groups[test_idx])
overlap = train_knife_set & test_knife_set
print(f"\n✓ Train-Test bıçak kesişimi: {len(overlap)} bıçak")
if len(overlap) == 0:
    print("  ✓ Mükemmel! Hiçbir bıçak hem train hem test'te yok.")
else:
    print("  ⚠ UYARI: Kesişim var, data leakage riski!")

# Train setinde sınıf dağılımı korundu mu?
print(f"\n✓ Train setinde Ra3 sınıf dağılımı:")
train_dist = pd.Series(y_clf_train).value_counts().sort_index()
for cls, count in train_dist.items():
    pct = count / len(y_clf_train) * 100
    print(f"  Sınıf {cls}: {count:5d} ({pct:.1f}%)")


# =============================================================================
# ADIM 6: CROSS-VALIDATION KURULUMU
# =============================================================================
# Test seti sadece en sonda bir kez kullanılır (final değerlendirme).
# Model geliştirme sırasında TRAIN seti içinde cross-validation yaparız.
#
# GroupKFold ne yapar?
#   → Train setini K parçaya böler.
#   → Her seferinde 1 parça validation, geri kalan K-1 parça eğitim olur.
#   → Yine grup bazlı: aynı bıçak validation'a ve train'e ikiye bölünmez.
#   → n_splits=5 → 5-fold CV → her bıçak 1 kez validation'da görünür.
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 6: Cross-Validation Kurulumu")
print("=" * 60)

cv = GroupKFold(n_splits=5)
train_groups = groups[train_idx]

print(f"✓ GroupKFold(n_splits=5) hazır.")
print(f"  Her fold'da yaklaşık {train_knives // 5} bıçak validation'da olacak.")
print(f"  Kullanımı: cv.split(X_train, y_clf_train, groups=train_groups)")
print(f"\n  Örnek fold boyutları:")
for fold_num, (tr, val) in enumerate(cv.split(X_train, y_clf_train, groups=train_groups), 1):
    print(f"    Fold {fold_num}: train={len(tr)} satır, validation={len(val)} satır "
          f"({len(set(train_groups[val]))} bıçak)")


# =============================================================================
# ADIM 7: VERİYİ KAYDET
# =============================================================================
# İşlenmiş veriyi numpy .npz formatında kaydediyoruz.
# Bu sayede cart_model.py ve c45_model.py tekrar Excel okumak zorunda kalmaz.
# .npz = numpy'nin sıkıştırılmış çoklu array formatı.
# =============================================================================

print("\n" + "=" * 60)
print("ADIM 7: İşlenmiş Veri Kaydediliyor...")
print("=" * 60)

np.savez(
    "data/processed/split_data.npz",
    X_train=X_train,
    X_test=X_test,
    y_clf_train=y_clf_train,
    y_clf_test=y_clf_test,
    y_reg_train=y_reg_train,
    y_reg_test=y_reg_test,
    train_groups=train_groups,
    test_groups=groups[test_idx]
)

# Feature isimlerini de ayrıca kaydet (model yorumlama aşamasında lazım olacak)
pd.Series(FEATURE_COLS).to_csv("data/processed/feature_names.csv", index=False)

print("✓ data/processed/split_data.npz   — train/test array'leri")
print("✓ data/processed/feature_names.csv — 42 feature ismi")

print("\n" + "=" * 60)
print("VERİ HAZIRLIĞI TAMAMLANDI")
print("Sıradaki adım: cart_model.py")
print("=" * 60)
