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
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# KONFIGURASI PATH MODEL
# ================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model_svm.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.pkl")

# ================================================================
# KONSTANTA VALIDASI
# ================================================================
CONFIDENCE_THRESHOLD = 60.0  # Minimum confidence (%)
MIN_OBJECT_AREA = 5000       # Minimum area objek (pixels)
MIN_MASK_COVERAGE = 2.0      # Minimum coverage mask terhadap gambar (%)

# ================================================================
# CUSTOM CSS
# ================================================================
def load_custom_css():
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .success-box {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        .info-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        .error-box {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

# ================================================================
# FUNGSI PREPROCESSING
# ================================================================
def segment_hsv(image):
    """Segmentasi multi-warna cabai"""
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
        return image, 0
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    
    # Padding 30%
    padding_w = int(w * 0.3)
    padding_h = int(h * 0.3)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(image.shape[1] - x, w + 2 * padding_w)
    h = min(image.shape[0] - y, h + 2 * padding_h)
    
    return image[y:y+h, x:x+w], area

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

def validate_chili_object(mask, image_shape):
    """Validasi apakah objek yang tersegmentasi valid sebagai cabai"""
    # Hitung coverage mask
    total_pixels = image_shape[0] * image_shape[1]
    mask_pixels = np.count_nonzero(mask)
    coverage = (mask_pixels / total_pixels) * 100
    
    # Hitung area objek terbesar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, 0, coverage, "Tidak ada objek terdeteksi"
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Validasi
    if area < MIN_OBJECT_AREA:
        return False, area, coverage, f"Objek terlalu kecil (area: {area:.0f} < {MIN_OBJECT_AREA})"
    
    if coverage < MIN_MASK_COVERAGE:
        return False, area, coverage, f"Coverage terlalu rendah ({coverage:.1f}% < {MIN_MASK_COVERAGE}%)"
    
    return True, area, coverage, "Valid"

def preprocess_image(image):
    """Pipeline preprocessing lengkap"""
    original_shape = image.shape
    
    # 1. Resize awal
    resized = cv2.resize(image, (664, 664))
    
    # 2. Segmentasi HSV
    mask = segment_hsv(resized)
    
    # 3. Morfologi
    morph = apply_morphology(mask)
    
    # 4. Validasi objek
    is_valid, area, coverage, message = validate_chili_object(morph, resized.shape)
    
    # 5. Apply mask
    masked = apply_mask_to_image(resized, morph)
    
    # 6. Crop objek dengan padding
    cropped, _ = crop_object_with_padding(masked, morph)
    
    # 7. Resize final dengan aspect ratio
    final = resize_with_padding(cropped, target_size=664)
    
    return final, mask, masked, cropped, is_valid, area, coverage, message

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
# FUNGSI VISUALISASI
# ================================================================
def plot_confusion_matrix(cm, classes):
    """Visualisasi confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Jumlah Prediksi'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig

def plot_class_distribution(y):
    """Visualisasi distribusi kelas"""
    unique, counts = np.unique(y, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#43e97b', '#f5576c', '#fa709a']
    bars = ax.bar(unique, counts, color=colors[:len(unique)])
    
    ax.set_xlabel('Kelas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jumlah Gambar', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Data Training', fontsize=16, fontweight='bold')
    ax.set_xticks(unique)
    ax.set_xticklabels([f'Kelas {i}' for i in unique])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def display_metrics_dashboard(metrics):
    """Tampilkan dashboard metrik"""
    st.markdown("### 📊 Dashboard Performa Model")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-label">Akurasi</div>
                <div class="metric-value">{metrics['accuracy']:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{metrics['precision']:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{metrics['recall']:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{metrics['f1_score']:.1%}</div>
            </div>
        """, unsafe_allow_html=True)

# ================================================================
# STREAMLIT UI
# ================================================================
st.set_page_config(
    page_title="Klasifikasi Kematangan Cabai", 
    page_icon="🌶️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌶️ Klasifikasi Kematangan Cabai</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Model SVM dengan Multi-Layer Validation
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png", width=100)
    st.markdown("### 📋 Menu Navigasi")
    menu = st.selectbox("Pilih Menu", ["🏠 Home", "🎓 Training", "🔮 Prediksi", "📁 Batch", "⚙️ Settings"])
    
    st.markdown("---")
    st.markdown("### 📌 Validasi Aktif")
    st.info(f"""
    ✅ Confidence ≥ {CONFIDENCE_THRESHOLD}%
    ✅ Area ≥ {MIN_OBJECT_AREA} px
    ✅ Coverage ≥ {MIN_MASK_COVERAGE}%
    """)
    
    if os.path.exists(METRICS_PATH):
        metrics = pickle.load(open(METRICS_PATH, "rb"))
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        st.success(f"**Akurasi: {metrics['accuracy']:.1%}**")

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
    0: "🟢", 1: "🔴", 2: "🟠"
}

# ================================================================
# MENU HOME
# ================================================================
if menu == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 👋 Selamat Datang!")
        st.markdown("""
        Aplikasi ini menggunakan **Machine Learning (SVM)** dengan **Multi-Layer Validation** 
        untuk mengklasifikasikan tingkat kematangan cabai.
        
        ### 🛡️ Sistem Validasi 3 Layer:
        """)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="info-box">
                <h4>Layer 1: Segmentasi</h4>
                <p>Deteksi objek cabai</p>
                <small>Min area: {MIN_OBJECT_AREA} px</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="warning-box">
                <h4>Layer 2: Coverage</h4>
                <p>Validasi ukuran objek</p>
                <small>Min: {MIN_MASK_COVERAGE}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div class="success-box">
                <h4>Layer 3: Confidence</h4>
                <p>Keyakinan prediksi</p>
                <small>Min: {CONFIDENCE_THRESHOLD}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Kelas Prediksi:")
        col1a, col2a, col3a = st.columns(3)
        with col1a:
            st.success("🟢 **Belum Matang**")
        with col2a:
            st.error("🔴 **Matang**")
        with col3a:
            st.warning("🟠 **Kematangan**")
    
    with col2:
        st.markdown("### 🔬 Teknologi")
        st.code("""
• Python 3.x
• Streamlit
• OpenCV
• Scikit-learn
• SVM Algorithm
• HSV Segmentation
• GLCM Features
• Multi-Layer Validation
        """, language="text")

# ================================================================
# MENU TRAINING
# ================================================================
elif menu == "🎓 Training":
    st.markdown("## 📦 Upload Dataset Training")
    
    st.markdown("""
    <div class="info-box">
        <strong>📁 Format Dataset:</strong><br>
        Upload file ZIP dengan struktur folder per kelas (belum matang, matang, kematangan)
    </div>
    """, unsafe_allow_html=True)

    zip_file = st.file_uploader("Upload file ZIP", type=["zip"])

    if zip_file is not None:
        with st.spinner("📦 Mengekstrak ZIP..."):
            if os.path.exists("dataset"):
                shutil.rmtree("dataset")

            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("dataset")

        st.success("✅ ZIP berhasil diekstrak!")

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔄 Preprocessing & ekstraksi fitur..."):
            data = []
            labels = []
            total_images = 0
            skipped = 0
            
            for folder in os.listdir("dataset"):
                folder_path = os.path.join("dataset", folder)
                if os.path.isdir(folder_path):
                    total_images += len([f for f in os.listdir(folder_path) 
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            processed = 0
            
            for folder in os.listdir("dataset"):
                folder_path = os.path.join("dataset", folder)

                if os.path.isdir(folder_path):
                    st.write(f"📂 Memproses folder: **{folder}**")
                    
                    for img_name in os.listdir(folder_path):
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            continue
                            
                        img_path = os.path.join(folder_path, img_name)
                        
                        try:
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            
                            # Preprocessing
                            processed_img, _, _, _, is_valid, area, coverage, msg = preprocess_image(img)
                            
                            # Skip jika tidak valid
                            if not is_valid:
                                skipped += 1
                                st.warning(f"⚠️ {img_name}: {msg}")
                                continue
                            
                            # Ekstraksi fitur
                            feat = extract_features(processed_img)
                            data.append(feat)
                            labels.append(folder)
                            
                            processed += 1
                            progress_bar.progress(processed / total_images)
                            status_text.text(f"✨ Valid: {processed} | Skip: {skipped} | Total: {total_images}")
                            
                        except Exception as e:
                            skipped += 1
                            st.warning(f"⚠️ {img_name}: {str(e)}")

        status_text.empty()
        progress_bar.empty()
        
        if len(data) == 0:
            st.error("❌ Tidak ada gambar valid untuk training!")
            st.stop()
        
        df = pd.DataFrame(data)
        df["label"] = labels

        unique_classes = df["label"].unique()
        
        if len(unique_classes) < 2:
            st.error(f"❌ Dataset hanya memiliki {len(unique_classes)} kelas!")
            st.stop()

        # Konversi label
        df["label_original"] = df["label"]
        df["label"] = df["label"].str.lower().map(label_map_str_to_int)
        
        if df["label"].isnull().any():
            st.warning("⚠️ Beberapa label tidak dikenali")
            unique_labels = df["label_original"].unique()
            auto_map = {label: idx for idx, label in enumerate(unique_labels)}
            df["label"] = df["label_original"].map(auto_map)

        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Ekstraksi Selesai!</h3>
            <p>Valid: <strong>{len(df)}</strong> | Skip: <strong>{skipped}</strong> | Kelas: <strong>{len(unique_classes)}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_dist = plot_class_distribution(df["label"].values)
            st.pyplot(fig_dist)
        
        with col2:
            st.write("**Preview Dataset:**")
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown("## 🤖 Training Model SVM")
        
        with st.spinner("⏳ Training..."):
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
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': acc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm
            }

        display_metrics_dashboard(metrics)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Confusion Matrix")
            class_names = [label_map_int_to_str[i] for i in sorted(df["label"].unique())]
            fig_cm = plot_confusion_matrix(cm, class_names)
            st.pyplot(fig_cm)
        
        with col2:
            st.markdown("### 📝 Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

        # Simpan model
        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(scaler, open(SCALER_PATH, "wb"))
        pickle.dump(metrics, open(METRICS_PATH, "wb"))

        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Training Selesai!</h3>
            <p><strong>Akurasi: {acc:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# MENU PREDIKSI
# ================================================================
elif menu == "🔮 Prediksi":
    st.markdown("## 📷 Upload Gambar Cabai")
    
    st.markdown(f"""
    <div class="info-box">
        🛡️ Sistem akan memvalidasi gambar dengan 3 layer:<br>
        • Layer 1: Area objek ≥ {MIN_OBJECT_AREA} pixels<br>
        • Layer 2: Coverage ≥ {MIN_MASK_COVERAGE}%<br>
        • Layer 3: Confidence ≥ {CONFIDENCE_THRESHOLD}%
    </div>
    """, unsafe_allow_html=True)

    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        img_array = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("❗ Model belum ditemukan. Silakan training terlebih dahulu.")
            st.stop()

        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))

        with st.spinner("🔄 Preprocessing..."):
            processed_img, mask, masked_img, cropped_img, is_valid, area, coverage, message = preprocess_image(img)

        st.markdown("### 📋 Tahapan Preprocessing")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="1️⃣ Original", use_column_width=True)
        
        with col2:
            st.image(mask, caption="2️             
                



