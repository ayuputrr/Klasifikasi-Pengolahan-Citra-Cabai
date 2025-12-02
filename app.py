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
# Gunakan path relatif untuk kompatibilitas deployment
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model_svm.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# ================================================================
# CUSTOM CSS UNTUK TAMPILAN MENARIK
# ================================================================
def load_custom_css():
    st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Metric cards */
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
        
        /* Success box */
        .success-box {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        /* Info box */
        .info-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        /* Warning box */
        .warning-box {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 1rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
        }
        
        /* Stacked metrics */
        .stacked-metrics {
            display: flex;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom button styling */
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
# FUNGSI VISUALISASI
# ================================================================
def plot_confusion_matrix(cm, classes):
    """Visualisasi confusion matrix dengan warna menarik"""
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
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def display_metrics_dashboard(metrics):
    """Tampilkan dashboard metrik dengan desain menarik"""
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

# Load custom CSS
load_custom_css()

# Header dengan gradient
st.markdown("""
    <div class="main-header">
        <h1>🌶️ Klasifikasi Kematangan Cabai</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
             Model SVM dengan Preprocessing Advanced: Resizing → Segmentasi HSV → Morfologi → Masking → Cropping Objek → Augmentasi Data → Konversi RGB ke HSV 
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar dengan styling
with st.sidebar:
    st.image("https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png", width=100)
    st.markdown("### 📋 Menu Navigasi")
    menu = st.selectbox("Pilih Menu", ["🏠 Home", "🎓 Ekstraksi + Training", "🔮 Prediksi Citra", "📁 Prediksi Batch"])
    
    st.markdown("---")
    st.markdown("### 📌 Info Aplikasi")
    st.info("""
    **Versi:** 2.0
    
    **Fitur:**
    - Training model custom
    - Prediksi real-time
    - Batch processing
    - Visualisasi lengkap
    """)
    
    # Tampilkan akurasi model jika ada
    if os.path.exists(METRICS_PATH):
        metrics = pickle.load(open(METRICS_PATH, "rb"))
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        st.success(f"**Akurasi: {metrics['accuracy']:.1%}**")
        st.metric("Precision", f"{metrics['precision']:.1%}")
        st.metric("Recall", f"{metrics['recall']:.1%}")
        st.metric("F1-Score", f"{metrics['f1_score']:.1%}")

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
# MENU HOME
# ================================================================
if menu == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 👋 Selamat Datang!")
        st.markdown("""
        Aplikasi ini menggunakan **Machine Learning (SVM)** untuk mengklasifikasikan tingkat kematangan cabai 
        berdasarkan citra digital dengan akurasi tinggi.
        
        ### 🎯 Fitur Utama:
        """)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            <div class="info-box">
                <h3>🎓 Training</h3>
                <p>Train model dengan dataset sendiri</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="success-box">
                <h3>🔮 Prediksi</h3>
                <p>Klasifikasi gambar cabai real-time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown("""
            <div class="warning-box">
                <h3>📁 Batch</h3>
                <p>Proses banyak gambar sekaligus</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Kelas Prediksi:")
        
        col1a, col2a, col3a = st.columns(3)
        with col1a:
            st.success("🟢 **Belum Matang** - Cabai hijau")
        with col2a:
            st.error("🔴 **Matang** - Cabai merah sempurna")
        with col3a:
            st.warning("🟠 **Kematangan** - Cabai merah kekuningan")
    
    with col2:
        st.markdown("### 🔬 Teknologi")
        st.code("""
        • Python 3.x
        • Streamlit
        • OpenCV
        • Scikit-learn
        • SVM Algorithm
        • HSV Color Space
        • GLCM Features
        """, language="text")
        
        st.markdown("### 📈 Pipeline")
        st.info("""
        1. **Upload** gambar/dataset
        2. **Preprocessing** (segmentasi HSV)
        3. **Feature extraction** (HSV + GLCM)
        4. **Training/Prediksi** (SVM)
        5. **Visualisasi** hasil
        """)

# ================================================================
# MENU TRAINING MODEL
# ================================================================
elif menu == "🎓 Ekstraksi + Training":
    st.markdown("## 📦 Upload Dataset Training")
    
    st.markdown("""
    <div class="info-box">
        <strong>📁 Format Dataset:</strong><br>
        Upload file ZIP dengan struktur folder per kelas (belum matang, matang, kematangan)
    </div>
    """, unsafe_allow_html=True)

    zip_file = st.file_uploader("Upload file ZIP", type=["zip"], help="Upload ZIP berisi folder per kelas")

    if zip_file is not None:
        with st.spinner("📦 Mengekstrak ZIP..."):
            # Hapus dataset lama
            if os.path.exists("dataset"):
                shutil.rmtree("dataset")

            # Ekstrak ZIP
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("dataset")

        st.success("✅ ZIP berhasil diekstrak!")

        # Mulai ekstraksi fitur
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔄 Melakukan preprocessing & ekstraksi fitur..."):
            data = []
            labels = []
            total_images = 0
            
            # Hitung total gambar
            for folder in os.listdir("dataset"):
                folder_path = os.path.join("dataset", folder)
                if os.path.isdir(folder_path):
                    total_images += len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
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
                            
                            # Preprocessing lengkap
                            processed_img, _, _, _ = preprocess_image(img)
                            
                            # Ekstraksi fitur
                            feat = extract_features(processed_img)
                            data.append(feat)
                            labels.append(folder)
                            
                            processed += 1
                            progress_bar.progress(processed / total_images)
                            status_text.text(f"✨ Diproses: {processed}/{total_images} gambar")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Gagal memproses {img_name}: {str(e)}")

        status_text.empty()
        progress_bar.empty()
        
        df = pd.DataFrame(data)
        df["label"] = labels

        # Cek jumlah kelas
        unique_classes = df["label"].unique()
        
        if len(unique_classes) < 2:
            st.error(f"❌ **Error: Dataset hanya memiliki {len(unique_classes)} kelas!**")
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

        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Ekstraksi Fitur Selesai!</h3>
            <p>Berhasil memproses <strong>{len(df)}</strong> gambar dari <strong>{len(unique_classes)}</strong> kelas</p>
        </div>
        """, unsafe_allow_html=True)

        # Visualisasi distribusi data
        st.markdown("### 📊 Distribusi Dataset")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_dist = plot_class_distribution(df["label"].values)
            st.pyplot(fig_dist)
        
        with col2:
            st.write("**Preview Dataset Fitur:**")
            st.dataframe(df.head(10), use_container_width=True)

        # ============================
        # TRAINING MODEL
        # ============================
        st.markdown("---")
        st.markdown("## 🤖 Training Model SVM")
        
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
            
            # Hitung metrik
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Simpan metrik
            metrics = {
                'accuracy': acc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'confusion_matrix': cm
            }

        # Tampilkan dashboard metrik
        display_metrics_dashboard(metrics)
        
        st.markdown("---")
        
        # Visualisasi hasil
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

        # Simpan Model, Scaler, & Metrics
        pickle.dump(model, open(MODEL_PATH, "wb"))
        pickle.dump(scaler, open(SCALER_PATH, "wb"))
        pickle.dump(metrics, open(METRICS_PATH, "wb"))

        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Training Selesai!</h3>
            <p>Model, Scaler, dan Metrics berhasil disimpan di <code>{MODEL_DIR}/</code></p>
            <p><strong>Akurasi Model: {acc:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# MENU PREDIKSI CITRA
# ================================================================
elif menu == "🔮 Prediksi Citra":
    st.markdown("## 📷 Upload Gambar Cabai untuk Prediksi")
    
    st.markdown("""
    <div class="info-box">
        💡 Model akan otomatis mengisolasi objek cabai dari background TANPA zoom berlebihan
    </div>
    """, unsafe_allow_html=True)

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
        with st.spinner("🔄 Melakukan preprocessing..."):
            processed_img, mask, masked_img, cropped_img = preprocess_image(img)

        # Tampilkan hasil preprocessing
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

        # Ekstraksi fitur
        with st.spinner("🔍 Mengekstrak fitur..."):
            feat = extract_features(processed_img)

        # Scaling & Prediksi
        feat_scaled = scaler.transform([feat])
        pred_label = model.predict(feat_scaled)[0]
        pred_proba = model.predict_proba(feat_scaled)[0]

        # Konversi label
        if isinstance(pred_label, str):
            pred_label_normalized = pred_label.lower()
            pred_label_int = label_map_str_to_int.get(pred_label_normalized, 0)
            pred_label_display = pred_label.title()
            emoji = label_colors.get(pred_label_normalized, "⚪")
        else:
            pred_label_int = int(pred_label)
            pred_label_display = label_map_int_to_str.get(pred_label_int, "Unknown")
            emoji = label_colors.get(pred_label_int, "⚪")

        # Hasil Prediksi dengan styling menarik
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")
        
        max_prob = np.max(pred_proba) * 100
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-size: 1.5rem;">
            <h2>{emoji} {pred_label_display}</h2>
            <p style="font-size: 2rem; margin: 1rem 0;">Confidence: {max_prob:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Probabilitas Per Kelas")
        
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
        
        # Visualisasi probabilitas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart dengan matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))
            colors_bar = ['#43e97b', '#f5576c', '#fa709a']
            bars = ax.barh(prob_labels, pred_proba * 100, color=colors_bar[:len(prob_labels)])
            ax.set_xlabel('Probabilitas (%)', fontweight='bold')
            ax.set_title('Distribusi Probabilitas Prediksi', fontweight='bold', fontsize=14)
            ax.set_xlim(0, 100)
            
            # Tambahkan nilai di ujung bar
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                        f'{pred_proba[i]*100:.1f}%',
                        ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            prob_df = pd.DataFrame({
                'Kelas': prob_labels,
                'Probabilitas': [f"{p*100:.2f}%" for p in pred_proba]
            })
            st.dataframe(prob_df, use_container_width=True)

# ================================================================
# MENU PREDIKSI BATCH
# ================================================================
elif menu == "📁 Prediksi Batch":
    st.markdown("## 📁 Prediksi Banyak Gambar Sekaligus")
    
    st.markdown("""
    <div class="info-box">
        💡 Upload ZIP berisi gambar-gambar cabai untuk prediksi batch
    </div>
    """, unsafe_allow_html=True)
    
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
                
                # Preprocessing
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
        progress_bar.empty()
        
        # Tampilkan hasil dalam tabel
        st.markdown("### 📊 Hasil Prediksi Batch")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Statistik hasil
        st.markdown("### 📈 Statistik Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        total_success = len([r for r in results if r['Status'] == '✅ Berhasil'])
        total_failed = len(results) - total_success
        success_rate = (total_success / len(results)) * 100 if len(results) > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-label">Berhasil</div>
                <div class="metric-value">{total_success}/{len(results)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="metric-label">Gagal</div>
                <div class="metric-value">{total_failed}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{success_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribusi prediksi
        if total_success > 0:
            st.markdown("### 📊 Distribusi Kelas Prediksi")
            
            # Hitung distribusi
            prediction_counts = {}
            for r in results:
                if r['Status'] == '✅ Berhasil':
                    pred = r['Prediksi'].split(' ', 1)[1] if ' ' in r['Prediksi'] else r['Prediksi']
                    prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Chart dengan matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(prediction_counts.keys())
                counts = list(prediction_counts.values())
                colors = ['#43e97b', '#f5576c', '#fa709a']
                
                bars = ax.bar(classes, counts, color=colors[:len(classes)])
                ax.set_xlabel('Kelas', fontsize=12, fontweight='bold')
                ax.set_ylabel('Jumlah', fontsize=12, fontweight='bold')
                ax.set_title('Distribusi Hasil Prediksi', fontsize=14, fontweight='bold')
                
                # Tambahkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                dist_df = pd.DataFrame({
                    'Kelas': classes,
                    'Jumlah': counts,
                    'Persentase': [f"{(c/total_success)*100:.1f}%" for c in counts]
                })
                st.dataframe(dist_df, use_container_width=True)
        
        # Download hasil sebagai CSV
        st.markdown("### 💾 Download Hasil")
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_batch.csv",
            mime="text/csv"
        )
        
        # Cleanup
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
