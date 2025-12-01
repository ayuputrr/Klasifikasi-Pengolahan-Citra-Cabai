import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
import albumentations as A

# ======================================================
# PATH INPUT & OUTPUT
# ======================================================
input_dir = r"C:/Users/ASUS/Downloads/datasetlabel"
output_dir = r"C:/Users/ASUS/Downloads/fix"

csv_output = os.path.join(output_dir, "dataset_fitur_augmented.csv")

labels_map = {
    "belum matang": 0,      # Cabai hijau segar
    "matang": 1,            # Cabai merah segar
    "kematangan": 2         # Cabai merah keriput
}

# ======================================================
# BUAT FOLDER PREPROCESSING
# ======================================================
subfolders = [
    "resize_awal", "hsv", "masking", "morfologi",
    "masked_image", "crop", "resize_final", "augmented"
]

for sf in subfolders:
    for lbl in labels_map.keys():
        os.makedirs(os.path.join(output_dir, sf, lbl), exist_ok=True)

# ======================================================
# RESIZE DENGAN ASPECT RATIO - TANPA DISTORSI
# ======================================================
def resize_with_padding(image, target_size=664):
    """
    Resize gambar dengan mempertahankan aspect ratio.
    Tambahkan padding hitam agar ukuran akhir 664x664.
    KUNCI UTAMA: Objek tidak akan ter-zoom dan tidak ada distorsi!
    """
    h, w = image.shape[:2]
    
    # Hitung scale factor (pilih yang terkecil agar fit)
    scale = min(target_size / w, target_size / h)
    
    # Resize dengan aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Buat canvas hitam 664x664
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Letakkan gambar di tengah canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

# ======================================================
# FUNGSI SEGMENTASI HSV - MULTI WARNA CABAI
# ======================================================
def segment_hsv(image):
    """
    Segmentasi cabai dengan berbagai tingkat kematangan:
    - Hijau (belum matang)
    - Hijau kekuningan (transisi)
    - Merah (matang)
    - Merah kekuningan (kematangan)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1. HIJAU - untuk cabai belum matang
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # 2. HIJAU KEKUNINGAN - transisi ke matang
    lower_yellow_green = np.array([25, 40, 40])
    upper_yellow_green = np.array([35, 255, 255])
    mask_yellow_green = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
    
    # 3. MERAH - cabai matang segar
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    
    # 4. MERAH KEKUNINGAN/ORANGE - cabai kematangan (sedikit keriput)
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Gabungkan semua mask
    mask_combined = mask_green + mask_yellow_green + mask_red + mask_orange
    
    return hsv, mask_combined

# ======================================================
# MORFOLOGI (OPENING + CLOSING) - HILANGKAN NOISE KAIN
# ======================================================
def apply_morphology(mask):
    """
    Opening: hilangkan noise kecil (titik-titik pada kain)
    Closing: tutup lubang pada objek cabai
    """
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    
    # Opening: hilangkan noise kecil
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing: tutup lubang pada objek
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    return closing

# ======================================================
# IMPLEMENT MASK KE GAMBAR (ISOLASI OBJEK)
# ======================================================
def apply_mask_to_image(image, mask):
    """
    Terapkan mask untuk mengisolasi objek cabai dari background.
    Background akan hitam, hanya objek cabai yang terlihat.
    """
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(image, mask_3ch)
    return masked_image

# ======================================================
# CROP KONTUR TERBESAR - DENGAN PADDING PROPORSIONAL
# ======================================================
def crop_object_with_padding(image, mask):
    """
    Crop objek dengan padding proporsional untuk menghindari zoom berlebihan.
    Objek akan tetap dalam ukuran relatif yang sama.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("    ⚠ Tidak ada kontur terdeteksi, gunakan gambar penuh")
        return image

    # Ambil kontur dengan area terbesar (objek cabai utama)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Padding 30% dari ukuran objek
    padding_w = int(w * 0.3)
    padding_h = int(h * 0.3)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(image.shape[1] - x, w + 2 * padding_w)
    h = min(image.shape[0] - y, h + 2 * padding_h)
    
    # Crop dengan padding
    cropped = image[y:y+h, x:x+w]
    
    return cropped

# ======================================================
# FUNGSI UNTUK KONVERSI NAMA FILE KE PNG
# ======================================================
def convert_to_png_filename(filename):
    """Konversi nama file ke format .png"""
    base_name = os.path.splitext(filename)[0]
    return base_name + ".png"

# ======================================================
# FUNGSI SIMPAN GAMBAR SEBAGAI PNG
# ======================================================
def save_as_png(image, path):
    """Simpan gambar dalam format PNG"""
    cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# ======================================================
# AUGMENTASI DATA - TANPA ZOOM
# ======================================================
def get_augmentation_pipeline():
    """
    Augmentasi TANPA rotasi untuk menghindari zoom sama sekali
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
    ])

def augment_image(image, num_augmentations=3):
    """Generate multiple augmented versions"""
    augmented_images = []
    transform = get_augmentation_pipeline()
    
    for _ in range(num_augmentations):
        augmented = transform(image=image)
        augmented_images.append(augmented['image'])
    
    return augmented_images

# ======================================================
# EKSTRAKSI FITUR HSV
# ======================================================
def extract_hsv_features(image):
    """
    Ekstrak fitur warna dari HSV.
    Fitur ini membedakan cabai hijau vs merah, serta tingkat kematangan.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    return [
        np.mean(H), np.std(H),      # Hue: warna dominan
        np.mean(S), np.std(S),      # Saturation: intensitas warna
        np.mean(V), np.std(V)       # Value: kecerahan
    ]

# ======================================================
# EKSTRAKSI FITUR GLCM (TEKSTUR)
# ======================================================
def extract_glcm_features(image):
    """
    Ekstrak fitur tekstur dari GLCM.
    Fitur ini membedakan cabai segar (halus) vs keriput.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True
    )

    feats = []
    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]

    for p in props:
        feats.append(graycoprops(glcm, p).mean())

    return feats

# ======================================================
# PROSES SEMUA GAMBAR + AUGMENTASI + SIMPAN FITUR
# ======================================================
data = []
NUM_AUGMENTATIONS = 3  # Jumlah augmentasi per gambar

print("\n" + "="*70)
print("MULAI PREPROCESSING & EKSTRAKSI FITUR (OUTPUT: PNG)")
print("="*70)

for lbl_name, lbl_value in labels_map.items():
    label_dir = os.path.join(input_dir, lbl_name)
    
    if not os.path.exists(label_dir):
        print(f"❌ Folder tidak ditemukan: {label_dir}")
        continue
    
    # Filter hanya file gambar
    files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📁 Memproses kelas '{lbl_name}' (label {lbl_value}): {len(files)} gambar")
    print("-" * 70)

    for idx, file in enumerate(files):
        img_path = os.path.join(label_dir, file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ❌ Gagal membaca: {file}")
            continue

        # Konversi nama file ke PNG
        png_filename = convert_to_png_filename(file)

        # ========== PREPROCESSING PIPELINE ==========
        
        # 1. Resize awal DENGAN ASPECT RATIO (NO DISTORTION!)
        resize1 = resize_with_padding(img, target_size=664)
        save_as_png(resize1, os.path.join(output_dir, "resize_awal", lbl_name, png_filename))

        # 2. Segmentasi HSV (setelah resize tanpa distorsi)
        hsv_img, mask = segment_hsv(resize1)
        save_as_png(hsv_img, os.path.join(output_dir, "hsv", lbl_name, png_filename))
        save_as_png(mask, os.path.join(output_dir, "masking", lbl_name, png_filename))

        # 3. Morfologi: Opening + Closing (hilangkan noise kain)
        morph = apply_morphology(mask)
        save_as_png(morph, os.path.join(output_dir, "morfologi", lbl_name, png_filename))

        # 4. Implement mask ke gambar (isolasi objek, background hitam)
        masked_img = apply_mask_to_image(resize1, morph)
        save_as_png(masked_img, os.path.join(output_dir, "masked_image", lbl_name, png_filename))

        # 5. Crop kontur terbesar dengan PADDING BESAR
        crop = crop_object_with_padding(masked_img, morph)
        save_as_png(crop, os.path.join(output_dir, "crop", lbl_name, png_filename))

        # 6. Resize final DENGAN ASPECT RATIO (NO ZOOM!)
        final_img = resize_with_padding(crop, target_size=664)
        save_as_png(final_img, os.path.join(output_dir, "resize_final", lbl_name, png_filename))

        # ========== EKSTRAKSI FITUR GAMBAR ASLI ==========
        hsv_feat = extract_hsv_features(final_img)
        glcm_feat = extract_glcm_features(final_img)
        fitur_all = [png_filename, lbl_value] + hsv_feat + glcm_feat
        data.append(fitur_all)

        # ========== AUGMENTASI ==========
        augmented_images = augment_image(final_img, NUM_AUGMENTATIONS)
        
        for aug_idx, aug_img in enumerate(augmented_images):
            # Nama file augmentasi dalam format PNG
            aug_filename = f"{os.path.splitext(png_filename)[0]}_aug{aug_idx}.png"
            
            # Simpan gambar augmentasi sebagai PNG
            save_as_png(aug_img, os.path.join(output_dir, "augmented", lbl_name, aug_filename))
            
            # Ekstraksi fitur dari gambar augmentasi
            hsv_feat_aug = extract_hsv_features(aug_img)
            glcm_feat_aug = extract_glcm_features(aug_img)
            fitur_aug = [aug_filename, lbl_value] + hsv_feat_aug + glcm_feat_aug
            data.append(fitur_aug)
        
        if (idx + 1) % 5 == 0 or (idx + 1) == len(files):
            print(f"  ✅ Progress: {idx + 1}/{len(files)} gambar (output: PNG)")

# ======================================================
# SIMPAN CSV
# ======================================================
columns = [
    "file_name", "label",
    "H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std",
    "contrast", "dissimilarity", "homogeneity",
    "energy", "correlation", "ASM"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(csv_output, index=False)

print("\n" + "="*70)
print("✅ PREPROCESSING & EKSTRAKSI FITUR SELESAI!")
print("="*70)
print(f"📊 Total data (asli + augmentasi): {len(df)} sampel")
print(f"💾 CSV tersimpan: {csv_output}")
print(f"🖼️  Format output: PNG (dengan kompresi optimal)")

# Tampilkan distribusi kelas
print("\n📈 Distribusi data per kelas:")
for lbl_name, lbl_value in labels_map.items():
    count = len(df[df['label'] == lbl_value])
    original = count // (NUM_AUGMENTATIONS + 1)
    augmented = count - original
    print(f"  • {lbl_name:15} (label {lbl_value}): {count:4} total "
          f"({original} asli + {augmented} augmentasi)")

print("\n" + "="*70)
print("PERBAIKAN YANG DITERAPKAN:")
print("  ✅ Resize awal DENGAN aspect ratio (tidak ada distorsi)")
print("  ✅ Crop dengan padding 30% (objek tidak terlalu besar)")
print("  ✅ Resize final dengan aspect ratio (tidak ada distorsi)")
print("  ✅ Padding hitam pada canvas (ukuran objek konsisten)")
print("  ✅ Augmentasi tanpa rotasi (tidak ada zoom sama sekali)")
print("  ✅ Semua output disimpan dalam format PNG")
print("="*70)
