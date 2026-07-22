# =============================================================================
# plot_surface_parameters_histogram.py (PNG & EPS SUPPORT)
# Plots the distribution of actual surface parameters across different lines
# Hochschule München / KIM — Erasmus+ Internship Project
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("📊 GENERATING SURFACE PARAMETERS DISTRIBUTION HISTOGRAM...")
print("=" * 100)

# Akademik Font Ayarı (Times New Roman)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# Veri setini yükleme yolları
data_path = "data/processed/split_data.npz"
raw_data_path = "data/raw/Kochmesser_ohne_prozessdaten.xlsx"

# Dosya kontrolü (Eğer Excel yoksa CSV versiyonuna bak)
if not os.path.exists(raw_data_path):
    csv_fallback = "Kochmesser_ohne_prozessdaten.xlsx - Kochmesser_ohne_prozessdaten.csv"
    if os.path.exists(csv_fallback):
        raw_data_path = csv_fallback
    else:
        print("[ERROR] Veri dosyası bulunamadı! Lütfen dosya yollarını kontrol edin.")
        exit()

# Veriyi oku
df = pd.read_excel(raw_data_path) if raw_data_path.endswith(".xlsx") else pd.read_csv(raw_data_path)

# 💡 HANGI PARAMETREYİ KIYASLAMAK İSTİYORSUNUZ? 
# Örnek olarak veri setindeki pürüzlülük kolonlarından birini seçiyoruz (Örn: 'Ra' veya 'Rz')
# Eğer senin kolon isimlerin farklıysa dökümandan bakıp burayı değiştirebilirsin kanka!
target_parameter = 'Ra' 

if target_parameter not in df.columns:
    # Eğer birebir eşleşme yoksa içinde pürüzlülük geçen ilk sayısal kolonu otomatik seçelim
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_cols = [c for c in numeric_cols if c not in ['Linie', 'Class', 'Sınıf', 'Knife_ID', 'ID']]
    if valid_cols:
        target_parameter = valid_cols[0]
    else:
        print("[ERROR] Kıyaslanacak anlamlı bir pürüzlülük parametresi bulunamadı!")
        exit()

print(f"✓ Selected Parameter for Comparison: '{target_parameter}'")

# --- GRAFİK ÇİZİMİ ---
plt.figure(figsize=(12, 6), dpi=300)

# Seçtiğimiz pürüzlülük parametresinin 10 farklı çizgi (Linie) üzerindeki dağılımını çiziyoruz
# Anlaşılır olması için en kritik çizgileri (perspektifleri) renkli dalgalar olarak kıyaslayalım
lines_to_plot = sorted(df['Linie'].unique())

for line_id in lines_to_plot:
    subset = df[df['Linie'] == line_id]
    # Yoğunluk grafiği (KDE - Histogram eğrisi) olarak üst üste bindiriyoruz
    sns.kdeplot(subset[target_parameter], label=f'Linie {line_id+1}', shade=True, alpha=0.15)

# Grafik Ayarları (Akademik Standartlarda)
plt.title(f"Surface Parameter '{target_parameter}' Distribution Across Measurement Perspectives", fontsize=12, fontweight='bold', pad=15)
plt.xlabel(f"Measured Value of {target_parameter}", fontsize=10, fontweight='bold')
plt.ylabel("Density (Frequency of Knives)", fontsize=10, fontweight='bold')

plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(title="Perspectives", loc="upper right", frameon=True)

plt.tight_layout()

# KLASÖRÜ KONTROL ET VE KAYDET (Hem PNG hem EPS) 🎯
output_fig_dir = os.path.join("outputs", "figures", "4_sensitivity_analysis")
os.makedirs(output_fig_dir, exist_ok=True)

save_png = os.path.join(output_fig_dir, "surface_parameter_distribution.png")
save_eps = os.path.join(output_fig_dir, "surface_parameter_distribution.eps")

plt.savefig(save_png, bbox_inches='tight', dpi=300)
plt.savefig(save_eps, format='eps', bbox_inches='tight')
plt.close()

