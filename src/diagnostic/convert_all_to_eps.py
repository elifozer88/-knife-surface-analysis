import os
import glob
from PIL import Image

print("=" * 80)
print("♻️ AUTOMATIC ACADEMIC EPS CONVERTER STARTED...")
print("=" * 80)

#Outputs klasöründeki tüm PNG grafiklerini bulur
png_files = glob.glob("outputs/**/*.png", recursive=True)

if not png_files:
    print("[WARNING] Klasörlerde hiç PNG grafik bulunamadı!")
else:
    success_count = 0
    for png_path in png_files:
        # EPS dosya yolunu oluştur
        eps_path = png_path.rsplit('.', 1)[0] + ".eps"
        
        try:
            # Görseli aç ve EPS olarak RGB modunda kaydet
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(eps_path, format="EPS")
                print(f"✓ Converted: '{png_path}' ➡️ '{eps_path}'")
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Could not convert {png_path}: {e}")
            
   