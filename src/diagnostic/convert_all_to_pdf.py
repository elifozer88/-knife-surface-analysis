import os
import glob
from PIL import Image

print("=" * 80)
print("♻️ AUTOMATIC ACADEMIC PDF CONVERTER STARTED...")
print("=" * 80)

# outputs klasörü altındaki tüm PNG grafiklerini bulur
png_files = glob.glob("outputs/**/*.png", recursive=True)

if not png_files:
    print("[WARNING] Klasörlerde dönüştürülecek hiç PNG görsel bulunamadı!")
else:
    success_count = 0
    for png_path in png_files:
        # PDF dosya yolunu oluştur (.png uzantısını .pdf yapar)
        pdf_path = png_path.rsplit('.', 1)[0] + ".pdf"
        
        try:
            # Görseli aç ve PDF olarak RGB modunda yüksek kalitede kaydet
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(pdf_path, format="PDF", quality=100)
                print(f"✓ Converted: '{png_path}' ➡️ '{pdf_path}'")
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Could not convert {png_path}: {e}")
 