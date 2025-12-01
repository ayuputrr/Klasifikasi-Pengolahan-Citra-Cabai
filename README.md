# 🌶️ Chili - Image Processing Pipeline

Sistem klasifikasi tingkat kematangan cabai menggunakan **Image Processing** dan **Machine Learning**. Proyek ini fokus pada **preprocessing citra** yang robust untuk menghasilkan dataset berkualitas tinggi tanpa distorsi.

---

## 📋 Deskripsi Proyek

### 🎯 Klasifikasi 3 Kategori Cabai:
- **Belum Matang** (Label 0): Cabai hijau segar
- **Matang** (Label 1): Cabai merah segar  
- **Kematangan** (Label 2): Cabai merah keriput

### 🔑 Fitur Utama:
- ✅ Preprocessing tanpa distorsi gambar
- ✅ Segmentasi HSV multi-warna untuk berbagai tingkat kematangan
- ✅ Morfologi untuk menghilangkan noise background
- ✅ Ekstraksi fitur warna (HSV) dan tekstur (GLCM)
- ✅ Augmentasi data otomatis (3x lipat)
- ✅ Output format PNG dengan kompresi optimal

---

## 🔄 Pipeline Pengolahan Citra

### **Diagram Alur:**
```
Input Image
    ↓
[1] Resize dengan Aspect Ratio (664×664 + padding)
    ↓
[2] Segmentasi HSV (Multi-range color masking)
    ↓
[3] Morfologi Opening + Closing (Noise removal)
    ↓
[4] Apply Mask (Isolasi objek, background hitam)
    ↓
[5] Crop dengan Padding 30% (Focus on object)
    ↓
[6] Resize Final dengan Aspect Ratio (664×664)
    ↓
[7] Augmentasi Data (Flip, Brightness, Noise, Blur)
    ↓
[8] Ekstraksi Fitur (HSV + GLCM)
    ↓
Output: PNG + CSV Dataset
```

---

## 🔍 Penjelasan Detail Setiap Tahap

### **1️⃣ Resize dengan Aspect Ratio**

**Fungsi:** `resize_with_padding(image, target_size=664)`

**Proses:**
- Hitung scale factor (pilih yang terkecil agar fit)
- Resize gambar dengan mempertahankan aspect ratio
- Buat canvas hitam 664×664 piksel
- Letakkan gambar di tengah canvas

**Keunggulan:**
- ✅ Tidak ada distorsi gambar
- ✅ Objek tidak ter-zoom berlebihan
- ✅ Aspect ratio tetap terjaga
- ✅ Padding hitam untuk standarisasi ukuran

**Output:** Folder `resize_awal/`

---

### **2️⃣ Segmentasi HSV Multi-Warna**

**Fungsi:** `segment_hsv(image)`

**Range Warna yang Dideteksi:**

| Warna | Hue Range | Saturation | Value | Kategori |
|-------|-----------|------------|-------|----------|
| 🟢 Hijau | 35-85° | 40-255 | 40-255 | Belum matang |
| 🟡 Hijau Kekuningan | 25-35° | 40-255 | 40-255 | Transisi |
| 🔴 Merah 1 | 0-10° | 70-255 | 50-255 | Matang segar |
| 🔴 Merah 2 | 170-180° | 70-255 | 50-255 | Matang segar |
| 🟠 Orange | 10-25° | 50-255 | 50-255 | Kematangan |

**Proses:**
```
RGB → HSV Conversion → Multiple Color Masks → Combine All Masks
```

**Keunggulan:**
- ✅ Mendeteksi semua tingkat kematangan cabai
- ✅ Robust terhadap variasi pencahayaan
- ✅ Memisahkan objek dari background (kain)

**Output:** Folder `hsv/` dan `masking/`

---

### **3️⃣ Morfologi (Opening + Closing)**

**Fungsi:** `apply_morphology(mask)`

**Operasi:**
- **Opening** (Erosi → Dilasi): Hilangkan noise kecil (titik-titik pada kain)
- **Closing** (Dilasi → Erosi): Tutup lubang pada objek cabai

**Parameter:**
- Opening: Kernel 5×5, 2 iterasi
- Closing: Kernel 9×9, 2 iterasi

**Visualisasi:**
```
Mask Kotor (noise banyak)
    ↓ Opening
Noise Hilang (objek tetap utuh)
    ↓ Closing
Lubang Tertutup (objek solid)
```

**Output:** Folder `morfologi/`

---

### **4️⃣ Apply Mask ke Gambar**

**Fungsi:** `apply_mask_to_image(image, mask)`

**Proses:**
- Konversi mask grayscale ke 3 channel (BGR)
- Operasi bitwise AND antara gambar asli dan mask
- Hasilnya: objek cabai terlihat, background hitam

**Keunggulan:**
- ✅ Isolasi objek dari background
- ✅ Background noise tereliminasi
- ✅ Fokus pada objek utama (cabai)

**Output:** Folder `masked_image/`

---

### **5️⃣ Crop dengan Padding Proporsional**

**Fungsi:** `crop_object_with_padding(image, mask)`

**Proses:**
- Deteksi kontur terbesar (objek cabai utama)
- Hitung bounding box dari kontur
- Tambahkan padding 30% dari ukuran objek
- Crop gambar sesuai bounding box + padding

**Keunggulan:**
- ✅ Fokus pada objek utama (cabai terbesar)
- ✅ Padding 30% mencegah objek terlalu besar
- ✅ Ukuran objek konsisten antar gambar
- ✅ Menghindari cropping terlalu ketat

**Output:** Folder `crop/`

---

### **6️⃣ Resize Final dengan Aspect Ratio**

**Fungsi:** `resize_with_padding(image, target_size=664)`

**Proses:** (sama dengan tahap 1)
- Resize hasil crop ke 664×664 dengan aspect ratio
- Tambahkan padding hitam untuk standarisasi

**Keunggulan:**
- ✅ Ukuran akhir konsisten (664×664)
- ✅ Tidak ada distorsi sama sekali
- ✅ Siap untuk training model

**Output:** Folder `resize_final/`

---

### **7️⃣ Augmentasi Data**

**Fungsi:** `augment_image(image, num_augmentations=3)`

**Teknik Augmentasi:**

| Teknik | Probability | Parameter |
|--------|-------------|-----------|
| Horizontal Flip | 50% | - |
| Vertical Flip | 50% | - |
| Random Brightness/Contrast | 60% | ±20% |
| Gaussian Noise | 30% | var_limit=(10, 30) |
| Blur | 20% | blur_limit=3 |

**❌ TIDAK ADA ROTASI** (menghindari zoom berlebihan)

**Keunggulan:**
- ✅ Meningkatkan jumlah data 4x lipat
- ✅ Meningkatkan robustness model
- ✅ Mencegah overfitting
- ✅ Variasi pencahayaan dan noise

**Output:** Folder `augmented/` (3 gambar per input)

---

### **8️⃣ Ekstraksi Fitur**

#### **A. Fitur HSV (Color Features)**

**Fungsi:** `extract_hsv_features(image)`

Membedakan tingkat kematangan berdasarkan warna:

| Fitur | Deskripsi | Kegunaan |
|-------|-----------|----------|
| `H_mean` | Rata-rata Hue | Warna dominan (hijau vs merah) |
| `H_std` | Standar deviasi Hue | Variasi warna pada objek |
| `S_mean` | Rata-rata Saturation | Intensitas/kepekatan warna |
| `S_std` | Standar deviasi Saturation | Keseragaman intensitas |
| `V_mean` | Rata-rata Value | Tingkat kecerahan |
| `V_std` | Standar deviasi Value | Variasi kecerahan |

**Total:** 6 fitur warna

---

#### **B. Fitur GLCM (Texture Features)**

**Fungsi:** `extract_glcm_features(image)`

Membedakan cabai segar vs keriput:

| Fitur | Deskripsi | Kegunaan |
|-------|-----------|----------|
| `contrast` | Perbedaan intensitas piksel tetangga | Deteksi kerutan/tekstur kasar |
| `dissimilarity` | Ketidakmiripan tekstur | Variasi permukaan |
| `homogeneity` | Keseragaman tekstur | Tinggi = halus, rendah = kasar |
| `energy` | Keseragaman pola tekstur | Konsistensi permukaan |
| `correlation` | Korelasi nilai piksel | Keteraturan pola |
| `ASM` | Angular Second Moment | Uniformitas distribusi intensitas |

**Parameter GLCM:**
- Distance: 1 pixel
- Angles: 0°, 45°, 90°, 135° (4 arah)
- Levels: 256 (grayscale)
- Symmetric: True
- Normed: True

**Total:** 6 fitur tekstur

---

## 📊 Output Dataset

### **Struktur Folder Output:**
```
fix/
├── resize_awal/       # Resize pertama (664x664 + padding)
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── hsv/               # Gambar dalam color space HSV
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── masking/           # Binary mask hasil segmentasi
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── morfologi/         # Mask setelah opening+closing
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── masked_image/      # Gambar dengan background hitam
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── crop/              # Hasil cropping objek + padding
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
├── resize_final/      # Resize final (664x664 + padding)
│   ├── belum matang/
│   ├── matang/
│   └── kematangan/
└── augmented/         # Gambar hasil augmentasi
    ├── belum matang/
    ├── matang/
    └── kematangan/
```

---

### **File CSV: `dataset_fitur_augmented.csv`**

| Column | Tipe | Deskripsi |
|--------|------|-----------|
| `file_name` | String | Nama file gambar (format PNG) |
| `label` | Integer | 0=belum matang, 1=matang, 2=kematangan |
| `H_mean` | Float | Rata-rata Hue (warna dominan) |
| `H_std` | Float | Standar deviasi Hue |
| `S_mean` | Float | Rata-rata Saturation (intensitas warna) |
| `S_std` | Float | Standar deviasi Saturation |
| `V_mean` | Float | Rata-rata Value (kecerahan) |
| `V_std` | Float | Standar deviasi Value |
| `contrast` | Float | GLCM - Contrast (kerutan) |
| `dissimilarity` | Float | GLCM - Dissimilarity |
| `homogeneity` | Float | GLCM - Homogeneity (kehalusan) |
| `energy` | Float | GLCM - Energy |
| `correlation` | Float | GLCM - Correlation |
| `ASM` | Float | GLCM - Angular Second Moment |

**Total:** 14 kolom (1 nama file + 1 label + 6 fitur HSV + 6 fitur GLCM)

---

## 🚀 Cara Penggunaan

### **1. Instalasi Dependencies**
```bash
pip install opencv-python numpy pandas scikit-image scikit-learn albumentations
```

### **2. Persiapan Dataset**

Struktur folder input:
```
datasetlabel/
├── belum matang/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── matang/
│   ├── img3.jpg
│   ├── img4.jpg
│   └── ...
└── kematangan/
    ├── img5.jpg
    ├── img6.jpg
    └── ...
```

### **3. Konfigurasi Path**

Edit bagian ini di script:
```python
input_dir = r"C:/Users/ASUS/Downloads/datasetlabel"  # Path dataset input
output_dir = r"C:/Users/ASUS/Downloads/fix"          # Path output
```

### **4. Jalankan Script**
```bash
python preprocessing.py
```

### **5. Output**

Setelah proses selesai, Anda akan mendapatkan:
- ✅ 8 folder preprocessing (resize_awal, hsv, masking, dll.)
- ✅ File CSV: `dataset_fitur_augmented.csv`
- ✅ Total data: **N_asli × 4** (1 asli + 3 augmentasi)

**Contoh Output Console:**
```
======================================================================
MULAI PREPROCESSING & EKSTRAKSI FITUR (OUTPUT: PNG)
======================================================================

📁 Memproses kelas 'belum matang' (label 0): 100 gambar
----------------------------------------------------------------------
  ✅ Progress: 100/100 gambar (output: PNG)

📁 Memproses kelas 'matang' (label 1): 100 gambar
----------------------------------------------------------------------
  ✅ Progress: 100/100 gambar (output: PNG)

📁 Memproses kelas 'kematangan' (label 2): 100 gambar
----------------------------------------------------------------------
  ✅ Progress: 100/100 gambar (output: PNG)

======================================================================
✅ PREPROCESSING & EKSTRAKSI FITUR SELESAI!
======================================================================
📊 Total data (asli + augmentasi): 1200 sampel
💾 CSV tersimpan: C:/Users/ASUS/Downloads/fix/dataset_fitur_augmented.csv
🖼️  Format output: PNG (dengan kompresi optimal)

📈 Distribusi data per kelas:
  • belum matang   (label 0):  400 total (100 asli + 300 augmentasi)
  • matang         (label 1):  400 total (100 asli + 300 augmentasi)
  • kematangan     (label 2):  400 total (100 asli + 300 augmentasi)
```

---

## ✨ Keunggulan Pipeline Ini

### **Aspek Teknis:**

1. **✅ No Distortion**
   - Resize dengan aspect ratio + padding
   - Objek tidak ter-stretch atau ter-squash
   - Bentuk asli cabai tetap terjaga

2. **✅ No Excessive Zoom**
   - Cropping dengan padding proporsional 30%
   - Objek tidak terlalu besar dalam frame
   - Ukuran relatif konsisten antar gambar

3. **✅ Robust Segmentation**
   - Multi-range HSV untuk semua tingkat kematangan
   - Hijau, kuning, merah, orange terdeteksi
   - Toleran terhadap variasi pencahayaan

4. **✅ Noise Removal**
   - Morfologi opening-closing hilangkan noise kain
   - Objek cabai tetap utuh dan solid
   - Background bersih tanpa artefak

5. **✅ Consistent Size**
   - Semua gambar output 664×664 piksel
   - Padding hitam untuk standarisasi
   - Siap untuk input model deep learning

6. **✅ PNG Format**
   - Kompresi lossless untuk kualitas maksimal
   - Ukuran file lebih kecil dari BMP
   - Kompatibel dengan semua framework ML

---

### **Aspek Machine Learning:**

1. **Dataset Seimbang**
   - Augmentasi otomatis per kelas
   - Mencegah class imbalance
   - Meningkatkan generalisasi model

2. **Fitur Komprehensif**
   - 6 fitur warna (HSV) → membedakan kematangan
   - 6 fitur tekstur (GLCM) → membedakan kesegaran
   - Total 12 fitur numerik siap training

3. **Format Siap Pakai**
   - CSV terstruktur dengan header jelas
   - Label numerik (0, 1, 2)
   - Dapat langsung di-load ke pandas/sklearn

4. **Preprocessing Reproducible**
   - Semua parameter terdokumentasi
   - Proses dapat diulang dengan hasil konsisten
   - Mudah di-scale untuk dataset lebih besar

---

## 🔧 Customization

### **Ubah Jumlah Augmentasi:**
```python
NUM_AUGMENTATIONS = 5  # Default: 3
```

### **Ubah Ukuran Output:**
```python
resize_with_padding(image, target_size=512)  # Default: 664
```

### **Ubah Range Warna HSV:**
```python
# Contoh: lebih sensitif terhadap hijau muda
lower_green1 = np.array([30, 30, 30])  # Lebih rendah
upper_green1 = np.array([90, 255, 255])  # Lebih tinggi
```

### **Ubah Parameter Morfologi:**
```python
kernel_open = np.ones((7, 7), np.uint8)   # Default: (5,5)
kernel_close = np.ones((11, 11), np.uint8) # Default: (9,9)
```

### **Ubah Padding Crop:**
```python
# Di fungsi crop_object_with_padding()
padding_w = int(w * 0.5)  # Default: 0.3 (30%)
padding_h = int(h * 0.5)
```

### **Tambah Teknik Augmentasi:**
```python
def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.GaussNoise(p=0.3),
        A.Blur(p=0.2),
        # Tambahan:
        A.HueSaturationValue(p=0.3),      # Variasi warna
        A.CLAHE(p=0.3),                   # Peningkatan kontras
        A.RandomGamma(p=0.3),             # Variasi gamma
    ])
```

---

## 📝 Catatan Penting

### ⚠️ **Hal yang Harus Diperhatikan:**

1. **Struktur Dataset Input**
   - Folder harus berlabel: `belum matang/`, `matang/`, `kematangan/`
   - Nama folder harus exact match (case-sensitive)
   - Format input: JPG, JPEG, atau PNG

2. **Spesifikasi Hardware**
   - RAM minimum: 4GB (untuk dataset <500 gambar)
   - RAM recommended: 8GB+ (untuk dataset >1000 gambar)
   - Storage: ~2-3x ukuran dataset asli

3. **Waktu Proses**
   - ~2-5 detik per gambar (tergantung spesifikasi PC)
   - Contoh: 300 gambar = ~15-25 menit
   - Gunakan SSD untuk proses lebih cepat

4. **Format Output**
   - Semua output PNG (tidak ada JPG)
   - Kompresi PNG level 9 (maksimal)
   - Nama file original dipertahankan

---

### 🎯 **Best Practice:**

1. **Kualitas Gambar Input**
   - Resolusi minimal: 800×600 piksel
   - Resolusi recommended: 1920×1080 atau lebih tinggi
   - Format JPG/PNG dengan kualitas baik

2. **Background dan Pencahayaan**
   - Background kontras dengan objek (kain putih/hitam ideal)
   - Pencahayaan merata tanpa bayangan keras
   - Hindari over-exposure atau under-exposure

3. **Posisi Objek**
   - Objek cabai jelas terlihat
   - Tidak tertutup tangan/objek lain
   - Posisi bebas (horizontal/vertikal/diagonal)

4. **Jumlah Dataset**
   - Minimum 50 gambar per kelas (total 150)
   - Recommended 100-200 gambar per kelas
   - Dengan augmentasi 3x, akan menjadi 200-800 gambar per kelas

---

## 🐛 Troubleshooting

### **Problem 1: "Tidak ada kontur terdeteksi"**
**Penyebab:** Segmentasi HSV gagal (warna objek tidak terdeteksi)

**Solusi:**
```python
# Coba perluas range HSV
lower_green1 = np.array([30, 30, 30])  # Lebih lebar
upper_green1 = np.array([90, 255, 255])
```

---

### **Problem 2: Background ikut terdeteksi**
**Penyebab:** Range warna terlalu lebar

**Solusi:**
```python
# Persempit range dan tingkatkan threshold Saturation
lower_green1 = np.array([35, 60, 60])  # Saturation minimal naik
upper_green1 = np.array([85, 255, 255])
```

---

### **Problem 3: Objek terpotong saat crop**
**Penyebab:** Padding terlalu kecil

**Solusi:**
```python
# Tingkatkan padding
padding_w = int(w * 0.5)  # Dari 0.3 ke 0.5 (50%)
padding_h = int(h * 0.5)
```

---

### **Problem 4: Gambar terlalu gelap/terang**
**Penyebab:** Pencahayaan input tidak merata

**Solusi:**
```python
# Tambahkan preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization)
import cv2
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)
image = cv2.merge([l, a, b])
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
```

---

### **Problem 5: Memory Error**
**Penyebab:** Dataset terlalu besar untuk RAM

**Solusi:**
- Proses batch per kelas (1 kelas sekali)
- Resize input lebih kecil: `target_size=512` atau `448`
- Kurangi jumlah augmentasi: `NUM_AUGMENTATIONS = 1`

---

## 📚 Dependensi Library

| Library | Versi | Kegunaan |
|---------|-------|----------|
| OpenCV | 4.5+ | Operasi citra (resize, morfologi, color space) |
| NumPy | 1.19+ | Operasi array dan matriks |
| Pandas | 1.2+ | Manajemen dataset tabular (CSV) |
| scikit-image | 0.18+ | Ekstraksi fitur GLCM |
| scikit-learn | 0.24+ | StandardScaler (preprocessing fitur) |
| Albumentations | 1.0+ | Library augmentasi modern |

**Install semua:**
```bash
pip install opencv-python==4.8.1.78 numpy==1.24.3 pandas==2.0.3 scikit-image==0.21.0 scikit-learn==1.3.0 albumentations==1.3.1
```

---

## 📈 Performa yang Diharapkan

### **Setelah Preprocessing:**

| Metrik | Target |
|--------|--------|
| Dataset Augmented | 4x dataset asli |
| Akurasi Segmentasi | >95% objek terdeteksi |
| Ukuran File PNG | ~200-500 KB per gambar |
| Waktu Proses | 2-5 detik per gambar |

### **Untuk Training Model:**

| Model | Akurasi yang Diharapkan |
|-------|-------------------------|
| SVM (RBF Kernel) | 85-92% |
| Random Forest | 80-88% |
| CNN (ResNet/EfficientNet) | 90-96% |

---

## 🔬 Penelitian Lanjutan

### **Kemungkinan Improvement:**

1. **Deep Learning End-to-End**
   - Transfer learning (ResNet, EfficientNet, MobileNet)
   - Tidak perlu ekstraksi fitur manual
   - Akurasi lebih tinggi (95%+)

2. **Real-Time Detection**
   - Implementasi YOLO/SSD untuk deteksi cabai
   - Klasifikasi + lokalisasi dalam satu model
   - Cocok untuk aplikasi mobile/embedded

3. **Multi-Task Learning**
   - Prediksi kematangan + estimasi berat
   - Deteksi defect (busuk, cacat)
   - Grading otomatis (A, B, C)

4. **Deployment**
   - Web app (Flask/FastAPI + React)
   - Mobile app (TensorFlow Lite)
   - Edge device (Raspberry Pi + camera)



---



**🌶️ Happy Preprocessing! Semoga berhasil dalam penelitian Anda!**

---

## 📊 Contoh Visualisasi Pipeline

### **Input → Output:**
```
[Input: img1.jpg - 1920x1080]
    ↓
[Resize: 664x664 dengan padding hitam]
    ↓
[Segmentasi: Mask hijau/merah/orange]
    ↓
[Morfologi: Noise hilang, objek solid]
    ↓
[Masking: Background hitam, objek terlihat]
    ↓
[Crop: Fokus objek + padding 30%]
    ↓
[Resize Final: 664x664 standar]
    ↓
[Output: img1.png + img1_aug0.png + img1_aug1.png + img1_aug2.png]
```

---



**Last Updated:** December 2025  
**Author  :** Maulina Ayu S




