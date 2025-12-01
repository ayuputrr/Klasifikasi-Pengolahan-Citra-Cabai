import streamlit as st
import zipfile
import shutil
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from PIL import Image

# ================================================================
# KONFIGURASI PATH MODEL
# ================================================================
# Gunakan path relatif untuk kompatibilitas deployment
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model_svm.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ================================================================
# FUNGSI PREPROCESSING (SAMA SEPERTI TRAINING - DENGAN PERBAIKAN)
# ================================================================
def segment_hsv(image):
    """Segmentasi multi-warna cabai: hijau, hijau kekuningan, merah, merah kekuningan"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Hijau
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Hijau kekuningan
    lower_yellow_green = np.array([25, 40, 40])
    upper_yellow_green = np.array([35, 255, 255])
    mask_yellow_green = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
    
    # Merah
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Merah kekuningan
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    return mask_green + mask_yellow_green + mask_red + mask_orange

def apply_morphology(mask):
    """Opening + Closing untuk hilangkan noise"""
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((9, 9), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return closing

def apply_mask_to_image(image, mask):
    """Isolasi objek dari background"""
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(image, mask_3ch)

def crop_object_with_padding(image, mask):
    """Crop objek dengan padding proporsional (30%)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image
    
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Padding 30% (proporsional)
    padding_w = int(w * 0.3)
    padding_h = int(h * 0.3)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(image.shape[1] - x, w + 2 * padding_w)
    h = min(image.shape[0] - y, h + 2 * padding_h)
    
    return image[y:y+h, x:x+w]

def resize_with_padding(image, target_size=664):
    """
    Resize dengan aspect ratio + padding hitam.
    KUNCI UTAMA: Objek tidak akan ter-zoom!
    """
    h, w = image.shape[:2]
    
    # Hitung scale factor
    scale = min(target_size / w, target_size / h)
    
    # Resize dengan aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas hitam 664x664
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Letakkan gambar di tengah
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def preprocess_image(image):
    """
    Pipeline preprocessing lengkap (TANPA ZOOM):
    resize → segment → morphology → mask → crop dengan padding → resize dengan aspect ratio
    """
    # 1. Resize awal
    resized = cv2.resize(image, (664, 664))
    
    # 2. Segmentasi HSV
    mask = segment_hsv(resized)
    
    # 3. Morfologi
    morph = apply_morphology(mask)
    
    # 4. Apply mask
    masked = apply_mask_to_image(resized, morph)
    
    # 5. Crop objek dengan padding proporsional (30%)
    cropped = crop_object_with_padding(masked, morph)
    
    # 6. Resize final dengan aspect ratio (NO ZOOM!)
    final = resize_with_padding(cropped, target_size=664)
    
    return final, mask, masked, cropped

# ================================================================
# FUNGSI EKSTRAKSI FITUR
# ================================================================
def extract_hsv_features(image):
    """Ekstrak fitur warna HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    return [
        np.mean(H), np.std(H),
        np.mean(S), np.std(S),
        np.mean(V), np.std(V)
    ]

def extract_glcm_features(image):
    """Ekstrak fitur tekstur GLCM"""
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
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    
    for p in props:
        feats.append(graycoprops(glcm, p).mean())
    
    return feats

def extract_features(image):
    """Ekstrak semua fitur (HSV + GLCM)"""
    hsv_feat = extract_hsv_features(image)
    glcm_feat = extract_glcm_features(image)
    return hsv_feat + glcm_feat

# ================================================================
# STREAMLIT UI
# ================================================================
st.set_page_config(page_title="Klasifikasi Kematangan Cabai", page_icon="🌶️", layout="wide")

st.title("🌶️ Aplikasi Klasifikasi Kematangan Cabai")
st.markdown("**Model SVM dengan Preprocessing Tanpa Zoom: Segmentasi HSV → Morfologi → Isolasi Objek → Resize dengan Aspect Ratio**")

menu = st.sidebar.selectbox("📋 Menu", ["Ekstraksi + Training", "Prediksi Citra", "Prediksi Batch (Folder)"])

# Mapping label
label_map_str_to_int = {
    "belum matang": 0,
    "matang": 1,
    "kematangan": 2
}

label_map_int_to_str = {
    0: "Belum Matang",
    1: "Matang",
    2: "Kematangan"
}

label_colors = {
    0: "🟢",
    1: "🔴",
    2: "🟠",
    "belum matang": "🟢",
    "matang": "🔴",
    "kematangan": "🟠",
    "Belum Matang": "🟢",
    "Matang": "🔴",
    "Kematangan": "🟠"
}

# ================================================================
# MENU TRAINING MODEL
# ================================================================
if menu == "Ekstraksi + Training":
    st.header("📦 Upload ZIP Dataset")
    st.info("📁 Struktur ZIP: folder per kelas (belum matang, matang, kematangan)")

    zip_file = st.file_uploader("Upload file ZIP", type=["zip"])

    if zip_file is not None:
        with st.spinner("Mengekstrak ZIP..."):
            # Hapus dataset lama
            if os.path.exists("dataset"):
                shutil.rmtree("dataset")

            # Ekstrak ZIP
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("dataset")

        st.success("✅ ZIP berhasil diekstrak!")

        # Mulai ekstraksi fitur
        with st.spinner("🔄 Melakukan preprocessing & ekstraksi fitur (tanpa zoom)..."):
            data = []
            labels = []

            for folder in os.listdir("dataset"):
                folder_path = os.path.join("dataset", folder)

                if os.path.isdir(folder_path):
                    st.write(f"📂 Memproses folder: **{folder}**")
                    
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)

                        try:
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            
                            # Preprocessing lengkap (TANPA ZOOM!)
                            processed_img, _, _, _ = preprocess_image(img)
                            
                            # Ekstraksi fitur
                            feat = extract_features(processed_img)
                            data.append(feat)
                            labels.append(folder)
                        except Exception as e:
                            st.warning(f"⚠️ Gagal memproses {img_name}: {str(e)}")

        df = pd.DataFrame(data)
        df["label"] = labels

        # Cek jumlah kelas
        unique_classes = df["label"].unique()
        
        if len(unique_classes) < 2:
            st.error(f"❌ **Error: Dataset hanya memiliki {len(unique_classes)} kelas!**")
            st.warning("⚠️ **Solusi:**")
            st.write("1. Pastikan ZIP berisi minimal 2 folder kelas berbeda")
            st.write("2. Contoh struktur ZIP yang benar:")
            st.code("""
dataset.zip/
├── belum matang/
│   ├── img1.jpg
│   └── img2.jpg
├── matang/
│   ├── img3.jpg
│   └── img4.jpg
└── kematangan/
    ├── img5.jpg
    └── img6.jpg
            """)
            st.info(f"📊 Kelas yang terdeteksi: {list(unique_classes)}")
            st.stop()

        # Konversi label string ke integer
        df["label_original"] = df["label"]
        df["label"] = df["label"].str.lower().map(label_map_str_to_int)
        
        # Handling label yang tidak dikenali
        if df["label"].isnull().any():
            st.warning("⚠️ Beberapa label tidak dikenali, menggunakan mapping otomatis")
            unique_labels = df["label_original"].unique()
            auto_map = {label: idx for idx, label in enumerate(unique_labels)}
            df["label"] = df["label_original"].map(auto_map)

        st.success(f"✅ Berhasil mengekstrak fitur dari **{len(df)}** gambar")
        st.info(f"📊 Jumlah kelas terdeteksi: **{len(unique_classes)}** → {list(unique_classes)}")
        st.write("**Preview Dataset Fitur:**")
        st.dataframe(df.head(10))

        # ============================
        # TRAINING MODEL
        # ============================
        st.header("🤖 Training Model SVM")
        
        with st.spinner("⏳ Training model..."):
            # Drop kolom yang tidak perlu
            X = df.drop(["label", "label_original"], axis=1, errors='ignore').values
            y = df["label"].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            # Model SVM
            model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
            model.fit(X_train, y_train)

            # Prediksi
            y_pred = model.predict(X_test)

        # Evaluasi
        st.subheader("📊 Evaluasi Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Confusion Matrix:")
            st.text(confusion_matrix(y_test, y_pred))
        
        with col2:
            acc = accuracy_score(y_test, y_pred)
            st.metric("Akurasi", f"{acc:.2%}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Simpan Model & Scaler
        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(scaler, open(SCALER_PATH, "wb"))

        st.success("✅ Model & Scaler berhasil disimpan!")
        st.info(f"📁 Lokasi: `{MODEL_DIR}/`")
        
        st.success("🎯 **Preprocessing yang diterapkan:**")
        st.write("✅ Crop dengan padding 30% (objek tidak terlalu besar)")
        st.write("✅ Resize dengan aspect ratio (tidak ada distorsi)")
        st.write("✅ Padding hitam pada canvas (ukuran objek konsisten)")
        st.write("✅ Objek TIDAK akan ter-zoom berlebihan!")


# ================================================================
# MENU PREDIKSI CITRA
# ================================================================
elif menu == "Prediksi Citra":
    st.header("📷 Upload Gambar Cabai untuk Prediksi")
    st.info("💡 Model akan otomatis mengisolasi objek cabai dari background TANPA zoom berlebihan")

    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        # Load gambar
        img_array = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Load Model
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("❗ Model belum ditemukan. Silakan training model terlebih dahulu.")
            st.stop()

        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))

        # Preprocessing
        with st.spinner("🔄 Melakukan preprocessing (tanpa zoom)..."):
            processed_img, mask, masked_img, cropped_img = preprocess_image(img)

        # Tampilkan hasil preprocessing
        st.subheader("📋 Tahapan Preprocessing (Tanpa Zoom)")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="1. Gambar Asli", use_column_width=True)
        
        with col2:
            st.image(mask, caption="2. Mask (Segmentasi)", use_column_width=True)
        
        with col3:
            st.image(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), caption="3. Objek Terpisah", use_column_width=True)
        
        with col4:
            st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="4. Crop + Padding 30%", use_column_width=True)
        
        with col5:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="5. Hasil Akhir (No Zoom!)", use_column_width=True)

        # Ekstraksi fitur
        with st.spinner("🔍 Mengekstrak fitur..."):
            feat = extract_features(processed_img)

        # Scaling & Prediksi
        feat_scaled = scaler.transform([feat])
        pred_label = model.predict(feat_scaled)[0]
        pred_proba = model.predict_proba(feat_scaled)[0]

        # Konversi label ke format yang konsisten
        if isinstance(pred_label, str):
            pred_label_normalized = pred_label.lower()
            pred_label_int = label_map_str_to_int.get(pred_label_normalized, 0)
            pred_label_display = pred_label.title()
            emoji = label_colors.get(pred_label_normalized, "⚪")
        else:
            pred_label_int = int(pred_label)
            pred_label_display = label_map_int_to_str.get(pred_label_int, "Unknown")
            emoji = label_colors.get(pred_label_int, "⚪")

        # Hasil Prediksi
        st.subheader("🎯 Hasil Prediksi")
        
        st.success(f"### {emoji} Kelas: **{pred_label_display}**")
        
        st.write("**Probabilitas per Kelas:**")
        
        # Buat dataframe probabilitas
        prob_labels = []
        for i, prob in enumerate(pred_proba):
            if hasattr(model, 'classes_'):
                class_label = model.classes_[i]
                if isinstance(class_label, str):
                    display_label = class_label.title()
                else:
                    display_label = label_map_int_to_str.get(class_label, f"Class {class_label}")
            else:
                display_label = label_map_int_to_str.get(i, f"Class {i}")
            prob_labels.append(display_label)
        
        prob_df = pd.DataFrame({
            'Kelas': prob_labels,
            'Probabilitas': [f"{p*100:.2f}%" for p in pred_proba]
        })
        st.dataframe(prob_df)
        
        # Bar chart probabilitas
        chart_data = pd.DataFrame({
            prob_labels[i]: [pred_proba[i]] for i in range(len(pred_proba))
        }).T
        st.bar_chart(chart_data)
        
        st.success("✅ **Objek cabai diproses dengan ukuran konsisten (NO ZOOM)!**")

# ================================================================
# MENU PREDIKSI BATCH (FOLDER)
# ================================================================
elif menu == "Prediksi Batch (Folder)":
    st.header("📁 Prediksi Banyak Gambar Sekaligus")
    st.info("💡 Upload ZIP berisi gambar-gambar cabai (tanpa subfolder) untuk prediksi batch")
    
    # Upload ZIP
    zip_file = st.file_uploader("Upload ZIP berisi gambar", type=["zip"], key="batch_zip")
    
    if zip_file is not None:
        # Load Model
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("❗ Model belum ditemukan. Silakan training model terlebih dahulu.")
            st.stop()
        
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        
        # Extract ZIP
        with st.spinner("📦 Mengekstrak ZIP..."):
            batch_dir = "batch_prediction"
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
            
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(batch_dir)
        
        st.success("✅ ZIP berhasil diekstrak!")
        
        # Cari semua file gambar
        image_files = []
        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if len(image_files) == 0:
            st.error("❌ Tidak ada gambar ditemukan dalam ZIP!")
            st.stop()
        
        st.info(f"📊 Ditemukan **{len(image_files)}** gambar untuk diprediksi")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Container untuk hasil
        results = []
        
        # Proses setiap gambar
        for idx, img_path in enumerate(image_files):
            status_text.text(f"🔄 Memproses: {os.path.basename(img_path)} ({idx+1}/{len(image_files)})")
            
            try:
                # Load gambar
                img = cv2.imread(img_path)
                if img is None:
                    results.append({
                        'Nama File': os.path.basename(img_path),
                        'Status': '❌ Gagal dibaca',
                        'Prediksi': '-',
                        'Confidence': '-'
                    })
                    continue
                
                # Preprocessing (TANPA ZOOM!)
                processed_img, _, _, _ = preprocess_image(img)
                
                # Ekstraksi fitur
                feat = extract_features(processed_img)
                
                # Prediksi
                feat_scaled = scaler.transform([feat])
                pred_label = model.predict(feat_scaled)[0]
                pred_proba = model.predict_proba(feat_scaled)[0]
                max_proba = np.max(pred_proba)
                
                # Konversi label
                if isinstance(pred_label, str):
                    pred_label_display = pred_label.title()
                    emoji = label_colors.get(pred_label.lower(), "⚪")
                else:
                    pred_label_display = label_map_int_to_str.get(int(pred_label), "Unknown")
                    emoji = label_colors.get(int(pred_label), "⚪")
                
                # Simpan hasil
                results.append({
                    'Nama File': os.path.basename(img_path),
                    'Status': '✅ Berhasil',
                    'Prediksi': f"{emoji} {pred_label_display}",
                    'Confidence': f"{max_proba*100:.2f}%"
                })
                
            except Exception as e:
                results.append({
                    'Nama File': os.path.basename(img_path),
                    'Status': f'❌ Error: {str(e)[:50]}',
                    'Prediksi': '-',
                    'Confidence': '-'
                })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(image_files))
        
        status_text.text("✅ Prediksi selesai!")
        
        # Tampilkan hasil dalam tabel
        st.subheader("📊 Hasil Prediksi Batch")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Statistik hasil
        st.subheader("📈 Statistik Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_success = len([r for r in results if r['Status'] == '✅ Berhasil'])
            st.metric("✅ Berhasil", f"{total_success}/{len(results)}")
        
        with col2:
            total_failed = len(results) - total_success
            st.metric("❌ Gagal", total_failed)
        
        with col3:
            success_rate = (total_success / len(results)) * 100 if len(results) > 0 else 0
            st.metric("📊 Success Rate", f"{success_rate:.1f}%")
        
        # Distribusi prediksi
        if total_success > 0:
            st.subheader("📊 Distribusi Kelas Prediksi")
            
            # Hitung distribusi
            prediction_counts = {}
            for r in results:
                if r['Status'] == '✅ Berhasil':
                    pred = r['Prediksi'].split(' ', 1)[1] if ' ' in r['Prediksi'] else r['Prediksi']
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            # Tampilkan dalam chart
            dist_df = pd.DataFrame({
                'Kelas': list(prediction_counts.keys()),
                'Jumlah': list(prediction_counts.values())
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.bar_chart(dist_df.set_index('Kelas'))
            
            with col2:
                st.dataframe(dist_df)
                for kelas, jumlah in prediction_counts.items():
                    pct = (jumlah / total_success) * 100
                    st.write(f"**{kelas}**: {jumlah} ({pct:.1f}%)")
        
        # Download hasil sebagai CSV
        st.subheader("💾 Download Hasil")
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_batch.csv",
            mime="text/csv"
        )
        
        st.success("✅ **Semua gambar diproses dengan preprocessing TANPA ZOOM!**")
        
        # Cleanup
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
