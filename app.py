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
import tempfile

# ================================================================
# KONFIGURASI PATH MODEL (MENGGUNAKAN TEMPDIR)
# ================================================================
MODEL_DIR = os.path.join(tempfile.gettempdir(), "chili_classifier_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model_svm.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ================================================================
# CUSTOM CSS UNTUK TAMPILAN LEBIH MENARIK
# ================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card style */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header style */
    h1 {
        color: #1e3a8a;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    h2, h3 {
        color: #1e40af;
    }
    
    /* Info box */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1e3a8a;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# FUNGSI PREPROCESSING
# ================================================================
def segment_hsv(image):
    """Segmentasi multi-warna cabai: hijau, hijau kekuningan, merah, merah kekuningan"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    lower_yellow_green = np.array([25, 40, 40])
    upper_yellow_green = np.array([35, 255, 255])
    mask_yellow_green = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
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
    
    padding_w = int(w * 0.3)
    padding_h = int(h * 0.3)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(image.shape[1] - x, w + 2 * padding_w)
    h = min(image.shape[0] - y, h + 2 * padding_h)
    
    return image[y:y+h, x:x+w]

def resize_with_padding(image, target_size=664):
    """Resize dengan aspect ratio + padding hitam"""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def preprocess_image(image):
    """Pipeline preprocessing lengkap"""
    resized = cv2.resize(image, (664, 664))
    mask = segment_hsv(resized)
    morph = apply_morphology(mask)
    masked = apply_mask_to_image(resized, morph)
    cropped = crop_object_with_padding(masked, morph)
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
st.set_page_config(
    page_title="Klasifikasi Kematangan Cabai", 
    page_icon="🌶️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header dengan styling
st.markdown("<h1>🌶️ Sistem Klasifikasi Kematangan Cabai</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 10px; background: white; border-radius: 10px; margin-bottom: 20px;'>
    <p style='color: #475569; font-size: 16px; margin: 0;'>
        <b>Teknologi:</b> SVM + HSV Segmentation + GLCM Features | <b>Preprocessing:</b> No Zoom, Aspect Ratio Preserved
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar menu dengan icon
st.sidebar.markdown("### 📋 Menu Navigasi")
menu = st.sidebar.radio(
    "",
    ["🎓 Ekstraksi + Training", "🔍 Prediksi Citra", "📁 Prediksi Batch"],
    label_visibility="collapsed"
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informasi")
st.sidebar.info(f"""
**Model Location:**  
`{MODEL_DIR}`

**Status Model:**  
{'✅ Tersedia' if os.path.exists(MODEL_PATH) else '❌ Belum Ditraining'}
""")

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
    0: "🟢", 1: "🔴", 2: "🟠",
    "belum matang": "🟢", "matang": "🔴", "kematangan": "🟠",
    "Belum Matang": "🟢", "Matang": "🔴", "Kematangan": "🟠"
}

# ================================================================
# MENU TRAINING MODEL
# ================================================================
if menu == "🎓 Ekstraksi + Training":
    st.markdown("## 📦 Upload Dataset untuk Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **📁 Struktur ZIP yang Dibutuhkan:**
        ```
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
        ```
        """)
    
    with col2:
        st.success("""
        **✨ Fitur Training:**
        - Auto extraction
        - HSV segmentation
        - GLCM features
        - SVM classifier
        """)

    zip_file = st.file_uploader("📤 Upload file ZIP dataset", type=["zip"], help="Upload ZIP berisi folder per kelas")

    if zip_file is not None:
        with st.spinner("⏳ Mengekstrak ZIP..."):
            if os.path.exists("dataset"):
                shutil.rmtree("dataset")

            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("dataset")

        st.success("✅ ZIP berhasil diekstrak!")

        with st.spinner("🔄 Melakukan preprocessing & ekstraksi fitur..."):
            data = []
            labels = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_folders = [f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))]
            
            for folder_idx, folder in enumerate(all_folders):
                folder_path = os.path.join("dataset", folder)
                status_text.text(f"📂 Memproses folder: {folder}")
                
                images = os.listdir(folder_path)
                for img_idx, img_name in enumerate(images):
                    img_path = os.path.join(folder_path, img_name)

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        processed_img, _, _, _ = preprocess_image(img)
                        feat = extract_features(processed_img)
                        data.append(feat)
                        labels.append(folder)
                    except Exception as e:
                        st.warning(f"⚠️ {img_name}: {str(e)[:50]}")
                
                progress_bar.progress((folder_idx + 1) / len(all_folders))

        df = pd.DataFrame(data)
        df["label"] = labels

        unique_classes = df["label"].unique()
        
        if len(unique_classes) < 2:
            st.error(f"❌ Dataset hanya memiliki {len(unique_classes)} kelas!")
            st.stop()

        df["label_original"] = df["label"]
        df["label"] = df["label"].str.lower().map(label_map_str_to_int)
        
        if df["label"].isnull().any():
            unique_labels = df["label_original"].unique()
            auto_map = {label: idx for idx, label in enumerate(unique_labels)}
            df["label"] = df["label_original"].map(auto_map)

        st.success(f"✅ Berhasil mengekstrak **{len(df)}** gambar dari **{len(unique_classes)}** kelas")
        
        with st.expander("📊 Preview Dataset Fitur"):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown("## 🤖 Training Model SVM")
        
        with st.spinner("⏳ Training model..."):
            X = df.drop(["label", "label_original"], axis=1, errors='ignore').values
            y = df["label"].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

        st.markdown("### 📊 Evaluasi Model")
        
        col1, col2, col3 = st.columns(3)
        
        acc = accuracy_score(y_test, y_pred)
        with col1:
            st.metric("🎯 Akurasi", f"{acc:.2%}", delta=f"{acc*100:.1f}%")
        
        with col2:
            st.metric("📈 Data Training", len(X_train))
        
        with col3:
            st.metric("📉 Data Testing", len(X_test))

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=[label_map_int_to_str.get(i, f"Class {i}") for i in range(len(cm))],
                                columns=[label_map_int_to_str.get(i, f"Class {i}") for i in range(len(cm))])
            st.dataframe(cm_df, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(scaler, open(SCALER_PATH, "wb"))

        st.success(f"✅ Model & Scaler berhasil disimpan di: `{MODEL_DIR}`")

# ================================================================
# MENU PREDIKSI CITRA
# ================================================================
elif menu == "🔍 Prediksi Citra":
    st.markdown("## 📷 Upload Gambar Cabai")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        img_file = st.file_uploader(
            "Pilih gambar cabai", 
            type=["jpg", "jpeg", "png"],
            help="Upload gambar cabai untuk klasifikasi"
        )
    
    with col2:
        st.info("""
        **💡 Tips:**
        - Gunakan foto yang jelas
        - Background kontras
        - Pencahayaan cukup
        """)

    if img_file is not None:
        img_array = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("❗ Model belum ditemukan. Silakan training model terlebih dahulu.")
            st.stop()

        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))

        with st.spinner("🔄 Memproses gambar..."):
            processed_img, mask, masked_img, cropped_img = preprocess_image(img)

        st.markdown("### 📋 Tahapan Preprocessing")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="1️⃣ Original", use_column_width=True)
        
        with col2:
            st.image(mask, caption="2️⃣ Mask", use_column_width=True)
        
        with col3:
            st.image(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB), caption="3️⃣ Isolated", use_column_width=True)
        
        with col4:
            st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="4️⃣ Cropped", use_column_width=True)
        
        with col5:
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="5️⃣ Final", use_column_width=True)

        with st.spinner("🔍 Mengekstrak fitur..."):
            feat = extract_features(processed_img)

        feat_scaled = scaler.transform([feat])
        pred_label = model.predict(feat_scaled)[0]
        pred_proba = model.predict_proba(feat_scaled)[0]

        if isinstance(pred_label, str):
            pred_label_normalized = pred_label.lower()
            pred_label_int = label_map_str_to_int.get(pred_label_normalized, 0)
            pred_label_display = pred_label.title()
            emoji = label_colors.get(pred_label_normalized, "⚪")
        else:
            pred_label_int = int(pred_label)
            pred_label_display = label_map_int_to_str.get(pred_label_int, "Unknown")
            emoji = label_colors.get(pred_label_int, "⚪")

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 40px; border-radius: 15px; text-align: center;'>
                <h1 style='color: white; margin: 0; font-size: 60px;'>{emoji}</h1>
                <h2 style='color: white; margin: 10px 0;'>{pred_label_display}</h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 18px;'>
                    Confidence: {np.max(pred_proba)*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📊 Probabilitas per Kelas:**")
            
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
            
            chart_data = pd.DataFrame({
                'Probabilitas': pred_proba
            }, index=prob_labels)
            
            st.bar_chart(chart_data, use_container_width=True)
            
            for i, label in enumerate(prob_labels):
                st.progress(pred_proba[i], text=f"{label}: {pred_proba[i]*100:.2f}%")

# ================================================================
# MENU PREDIKSI BATCH
# ================================================================
elif menu == "📁 Prediksi Batch":
    st.markdown("## 📁 Prediksi Batch (Multiple Images)")
    
    st.info("💡 Upload ZIP berisi gambar-gambar cabai (tanpa subfolder)")
    
    zip_file = st.file_uploader("📤 Upload ZIP", type=["zip"], key="batch")
    
    if zip_file is not None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("❗ Model belum tersedia. Training dulu ya!")
            st.stop()
        
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        
        with st.spinner("📦 Mengekstrak ZIP..."):
            batch_dir = "batch_prediction"
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
            
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(batch_dir)
        
        image_files = []
        for root, dirs, files in os.walk(batch_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if len(image_files) == 0:
            st.error("❌ Tidak ada gambar ditemukan!")
            st.stop()
        
        st.success(f"✅ Ditemukan **{len(image_files)}** gambar")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for idx, img_path in enumerate(image_files):
            status_text.text(f"🔄 {os.path.basename(img_path)} ({idx+1}/{len(image_files)})")
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    results.append({
                        'Nama File': os.path.basename(img_path),
                        'Status': '❌ Error',
                        'Prediksi': '-',
                        'Confidence': '-'
                    })
                    continue
                
                processed_img, _, _, _ = preprocess_image(img)
                feat = extract_features(processed_img)
                feat_scaled = scaler.transform([feat])
                pred_label = model.predict(feat_scaled)[0]
                pred_proba = model.predict_proba(feat_scaled)[0]
                max_proba = np.max(pred_proba)
                
                if isinstance(pred_label, str):
                    pred_label_display = pred_label.title()
                    emoji = label_colors.get(pred_label.lower(), "⚪")
                else:
                    pred_label_display = label_map_int_to_str.get(int(pred_label), "Unknown")
                    emoji = label_colors.get(int(pred_label), "⚪")
                
                results.append({
                    'Nama File': os.path.basename(img_path),
                    'Status': '✅ Success',
                    'Prediksi': f"{emoji} {pred_label_display}",
                    'Confidence': f"{max_proba*100:.2f}%"
                })
                
            except Exception as e:
                results.append({
                    'Nama File': os.path.basename(img_path),
                    'Status': f'❌ Error',
                    'Prediksi': '-',
                    'Confidence': '-'
                })
            
            progress_bar.progress((idx + 1) / len(image_files))
        
        status_text.text("✅ Selesai!")
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Prediksi Batch")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        total_success = len([r for r in results if r['Status'] == '✅ Success'])
        total_failed = len(results) - total_success
        success_rate = (total_success / len(results)) * 100 if len(results) > 0 else 0
        
        with col1:
            st.metric("✅ Berhasil", f"{total_success}/{len(results)}")
        
        with col2:
            st.metric("❌ Gagal", total_failed)
        
        with col3:
            st.metric("📊 Success Rate", f"{success_rate:.1f}%")
        
        if total_success > 0:
            st.markdown("### 📈 Distribusi Kelas")
            
            prediction_counts = {}
            for r in results:
                if r['Status'] == '✅ Success':
                    pred = r['Prediksi'].split(' ', 1)[1] if ' ' in r['Prediksi'] else r['Prediksi']
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                dist_df = pd.DataFrame({
                    'Jumlah': list(prediction_counts.values())
                }, index=list(prediction_counts.keys()))
                st.bar_chart(dist_df, use_container_width=True)
            
            with col2:
                for kelas, jumlah in prediction_counts.items():
                    pct = (jumlah / total_success) * 100
                    st.metric(kelas, f"{jumlah}", f"{pct:.1f}%")
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
        
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 20px;'>
    <p>🌶️ <b>Sistem Klasifikasi Kematangan Cabai</b> 🌶️</p>
    <p style='font-size: 14px; margin-top: 10px;'>
        Dibuat dengan ❤️ menggunakan Streamlit | SVM + HSV + GLCM Features
    </p>
    <p style='font-size: 12px; color: #94a3b8;'>
        © 2024 - Machine Learning Project
    </p>
</div>
""", unsafe_allow_html=True)
