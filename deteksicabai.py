import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings('ignore')

# KONFIGURASI
MODEL_PATH = r"C:\Users\Asus\downloads\bismillah\10_model_SVM\model_fixed.pkl"

# Page config
st.set_page_config(
    page_title="Sistem Prediksi Kematangan Cabai",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #27ae60;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #229954;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .confidence-bar {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

class BuahPredictor:
    def __init__(self, model_path):
        """Inisialisasi predictor"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.encoder = model_data['encoder']
        self.accuracy = model_data['accuracy']
    
    def remove_background(self, image):
        """Segmentasi otomatis untuk remove background"""
        # Cek brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 200:
            return image, True, "Background sudah bersih"
        
        # Otsu thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfologi
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Cari kontur terbesar
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
            
            result = np.ones_like(image) * 255
            result[mask_clean == 255] = image[mask_clean == 255]
            
            return result, True, "Segmentasi berhasil"
        else:
            return image, False, "Gagal deteksi objek"
    
    def extract_hsv_features(self, image, limited_mode=False):
        """Ekstraksi fitur HSV
        
        Args:
            image: Input image
            limited_mode: Jika True, hanya ekstrak mean dan std (10 fitur total)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        features = {}
        
        for i, channel_name in enumerate(['H', 'S', 'V']):
            channel = hsv[:, :, i]
            
            if np.sum(mask) > 0:
                channel_values = channel[mask > 0]
            else:
                channel_values = channel.flatten()
            
            # Ekstrak mean untuk semua channel
            features[f'mean_{channel_name}'] = np.mean(channel_values)
            features[f'std_{channel_name}'] = np.std(channel_values)
            
            if not limited_mode:
                # Ekstrak fitur tambahan jika tidak limited mode
                features[f'median_{channel_name}'] = np.median(channel_values)
                features[f'min_{channel_name}'] = np.min(channel_values)
                features[f'max_{channel_name}'] = np.max(channel_values)
        
        # Tambahkan 4 fitur lagi untuk mencapai 10 fitur jika limited_mode
        if limited_mode:
            # Hitung fitur tambahan dari RGB
            b, g, r = cv2.split(image)
            if np.sum(mask) > 0:
                r_values = r[mask > 0]
                g_values = g[mask > 0]
                b_values = b[mask > 0]
            else:
                r_values = r.flatten()
                g_values = g.flatten()
                b_values = b.flatten()
            
            features['mean_R'] = np.mean(r_values)
            features['mean_G'] = np.mean(g_values)
            features['mean_B'] = np.mean(b_values)
            features['brightness'] = np.mean(gray[mask > 0]) if np.sum(mask) > 0 else np.mean(gray)
        
        return features
    
    def extract_glcm_features(self, image):
        """Ekstraksi fitur GLCM menggunakan OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        gray_masked = gray.copy()
        gray_masked[mask == 0] = 0
        
        def compute_glcm(image, distance=1, angle=0):
            h, w = image.shape
            glcm = np.zeros((256, 256), dtype=np.float32)
            
            if angle == 0:
                dx, dy = distance, 0
            elif angle == 45:
                dx, dy = distance, -distance
            elif angle == 90:
                dx, dy = 0, -distance
            else:
                dx, dy = -distance, -distance
            
            for i in range(h):
                for j in range(w):
                    if 0 <= i+dy < h and 0 <= j+dx < w:
                        pixel1 = image[i, j]
                        pixel2 = image[i+dy, j+dx]
                        glcm[pixel1, pixel2] += 1
            
            if glcm.sum() > 0:
                glcm = glcm / glcm.sum()
            
            return glcm
        
        def glcm_properties(glcm):
            i, j = np.meshgrid(range(256), range(256), indexing='ij')
            
            contrast = np.sum(glcm * (i - j) ** 2)
            dissimilarity = np.sum(glcm * np.abs(i - j))
            homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
            energy = np.sum(glcm ** 2)
            asm = energy
            
            mean_i = np.sum(i * glcm)
            mean_j = np.sum(j * glcm)
            std_i = np.sqrt(np.sum(glcm * (i - mean_i) ** 2))
            std_j = np.sqrt(np.sum(glcm * (j - mean_j) ** 2))
            
            if std_i > 0 and std_j > 0:
                correlation = np.sum(glcm * (i - mean_i) * (j - mean_j)) / (std_i * std_j)
            else:
                correlation = 0
            
            return {
                'contrast': contrast,
                'dissimilarity': dissimilarity,
                'homogeneity': homogeneity,
                'energy': energy,
                'correlation': correlation,
                'ASM': asm
            }
        
        angles = [0, 45, 90, 135]
        all_props = {prop: [] for prop in ['contrast', 'dissimilarity', 'homogeneity', 
                                            'energy', 'correlation', 'ASM']}
        
        for angle in angles:
            glcm = compute_glcm(gray_masked, distance=1, angle=angle)
            props = glcm_properties(glcm)
            
            for prop_name, prop_value in props.items():
                all_props[prop_name].append(prop_value)
        
        features = {}
        for prop_name, prop_values in all_props.items():
            features[f'glcm_{prop_name}_mean'] = np.mean(prop_values)
            features[f'glcm_{prop_name}_std'] = np.std(prop_values)
        
        return features
    
    def predict(self, image):
        """Prediksi image"""
        # Segmentasi
        segmented, success, message = self.remove_background(image)
        
        # Cek jumlah fitur yang diharapkan
        n_features_expected = self.scaler.n_features_in_
        
        # Tentukan mode ekstraksi fitur berdasarkan jumlah yang diharapkan
        if n_features_expected == 10:
            # Mode 10 fitur: HSV limited mode (6) + RGB (3) + brightness (1)
            hsv_features = self.extract_hsv_features(segmented, limited_mode=True)
            all_features = hsv_features
            st.info("ℹ️ Menggunakan mode 10 fitur (HSV + RGB)")
        elif n_features_expected == 15:
            # Mode 15 fitur: HSV lengkap saja
            hsv_features = self.extract_hsv_features(segmented, limited_mode=False)
            all_features = hsv_features
            st.info("ℹ️ Menggunakan mode 15 fitur (HSV lengkap)")
        elif n_features_expected == 27:
            # Mode 27 fitur: HSV (15) + GLCM (12)
            hsv_features = self.extract_hsv_features(segmented, limited_mode=False)
            glcm_features = self.extract_glcm_features(segmented)
            all_features = {**hsv_features, **glcm_features}
            st.info("ℹ️ Menggunakan mode 27 fitur (HSV + GLCM)")
        else:
            # Mode custom: sesuaikan dengan feature names
            st.warning(f"⚠️ Mode custom: {n_features_expected} fitur")
            hsv_features = self.extract_hsv_features(segmented, limited_mode=False)
            glcm_features = self.extract_glcm_features(segmented)
            all_features = {**hsv_features, **glcm_features}
        
        # Debug info
        st.write(f"🔍 Fitur diekstrak: {len(all_features)}, Diharapkan: {n_features_expected}")
        
        # Jika jumlah fitur tidak sesuai, sesuaikan
        if len(all_features) != n_features_expected:
            # Ambil feature names yang diharapkan oleh model
            if hasattr(self.scaler, 'feature_names_in_'):
                expected_features = self.scaler.feature_names_in_
            else:
                # Jika tidak ada feature names, ambil n fitur pertama
                expected_features = list(all_features.keys())[:n_features_expected]
            
            with st.expander("⚠️ Penyesuaian Fitur"):
                st.write("**Fitur yang diharapkan:**", list(expected_features))
                st.write("**Fitur yang diekstrak:**", list(all_features.keys()))
            
            # Buat feature vector sesuai urutan yang diharapkan
            feature_values = []
            for feat_name in expected_features:
                if feat_name in all_features:
                    feature_values.append(all_features[feat_name])
                else:
                    feature_values.append(0.0)  # Default value jika fitur tidak ada
            
            feature_vector = np.array(feature_values).reshape(1, -1)
        else:
            feature_vector = np.array(list(all_features.values())).reshape(1, -1)
        
        # Prediksi
        feature_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_scaled)
        decision_scores = self.model.decision_function(feature_scaled)
        
        # Confidence
        confidence_normalized = np.exp(decision_scores[0]) / np.sum(np.exp(decision_scores[0])) * 100
        
        predicted_label = self.encoder.inverse_transform(prediction)[0]
        
        return {
            'predicted_class': predicted_label,
            'confidence_scores': {label: score for label, score in 
                                 zip(self.encoder.classes_, confidence_normalized)},
            'segmented_image': segmented,
            'segmentation_message': message,
            'features': all_features
        }

@st.cache_resource
def load_model():
    """Load model dengan caching"""
    try:
        predictor = BuahPredictor(MODEL_PATH)
        return predictor, None
    except Exception as e:
        return None, str(e)

def plot_confidence_chart(confidence_scores):
    """Plot confidence scores"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    labels = list(confidence_scores.keys())
    scores = list(confidence_scores.values())
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    bars = ax.barh(labels, scores, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=2)
    
    # Highlight highest
    max_idx = scores.index(max(scores))
    bars[max_idx].set_alpha(1.0)
    bars[max_idx].set_linewidth(3)
    
    # Labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        label_text = f'{score:.1f}%'
        if i == max_idx:
            label_text = f'★ {score:.1f}%'
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
               label_text, ha='left', va='center',
               fontweight='bold' if i == max_idx else 'normal',
               fontsize=12)
    
    ax.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Score per Kelas', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, 110])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(features, top_n=10):
    """Plot top N features"""
    # Sort features by value
    sorted_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(sorted_features.keys())
    values = list(sorted_features.values())
    
    colors_list = ['#3498db' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(names, values, color=colors_list, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Nilai Fitur', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {top_n} Fitur Terekstrak', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 30px;'>
            <h1 style='color: white; margin: 0;'>🌶️ Sistem Prediksi Kematangan Cabai</h1>
            <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>
                Klasifikasi Kematangan Cabai menggunakan Support Vector Machine (SVM)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor, error = load_model()
    
    if error:
        st.error(f"❌ Gagal memuat model: {error}")
        st.info("📝 Pastikan path model sudah benar dan model sudah ditraining")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/chili-pepper.png", width=100)
        st.title("📊 Informasi Model")
        
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Model Performance</h3>
            <p style='font-size: 32px; font-weight: bold; color: #27ae60; margin: 10px 0;'>
                {predictor.accuracy*100:.2f}%
            </p>
            <p style='color: #7f8c8d;'>Akurasi Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Kelas Prediksi")
        classes = predictor.encoder.classes_
        class_colors = {'belum matang': '🔴', 'matang': '🟠', 'kematangan': '🟢'}
        for cls in classes:
            emoji = class_colors.get(cls, '⚪')
            st.markdown(f"{emoji} **{cls.title()}**")
        
        st.markdown("---")
        
        st.markdown("### 📝 Cara Penggunaan")
        st.markdown("""
        1. Upload gambar buah
        2. Sistem akan otomatis:
           - Segmentasi background
           - Ekstraksi fitur
           - Prediksi kematangan
        3. Lihat hasil dan confidence score
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Fitur Sistem")
        st.markdown("""
        - ✅ Auto background removal
        - ✅ HSV color features
        - ✅ GLCM texture features
        - ✅ Real-time prediction
        - ✅ Confidence visualization
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Single", "📁 Prediksi Batch", "📖 Tentang"])
    
    with tab1:
        st.header("Upload Gambar Cabai")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Pilih gambar cabai (JPG, PNG, JPEG)",
                type=['jpg', 'jpeg', 'png'],
                help="Upload gambar cabai untuk diprediksi kematangannya",
                key="single_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    # Baca image dengan cara yang lebih robust
                    # Reset pointer file ke awal
                    uploaded_file.seek(0)
                    
                    # Baca sebagai PIL Image terlebih dahulu
                    pil_image = Image.open(uploaded_file)
                    
                    # Convert PIL ke numpy array
                    image_rgb = np.array(pil_image)
                    
                    # Convert RGB ke BGR untuk OpenCV
                    if len(image_rgb.shape) == 2:  # Grayscale
                        image = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
                    elif image_rgb.shape[2] == 4:  # RGBA
                        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
                    else:  # RGB
                        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Display original
                    st.subheader("📷 Gambar Asli")
                    st.image(image_rgb, 
                            caption=f"Ukuran: {image.shape[1]}x{image.shape[0]}", 
                            use_container_width=True)
                    
                    # Predict button
                    if st.button("🔮 Prediksi Sekarang!", type="primary", key="predict_btn"):
                        with st.spinner("⏳ Memproses gambar..."):
                            try:
                                result = predictor.predict(image)
                                
                                # Save to session state
                                st.session_state['result'] = result
                                st.session_state['original_image'] = image
                                st.success("✅ Prediksi berhasil!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error saat prediksi: {str(e)}")
                                st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error membaca gambar: {str(e)}")
                    st.info("💡 Tips: Pastikan file adalah gambar yang valid (JPG/PNG)")
        
        with col2:
            if 'result' in st.session_state:
                result = st.session_state['result']
                original_image = st.session_state['original_image']
                
                # Display segmented image
                st.subheader("🔧 Setelah Segmentasi")
                segmented_rgb = cv2.cvtColor(result['segmented_image'], cv2.COLOR_BGR2RGB)
                st.image(segmented_rgb,
                        caption=result['segmentation_message'],
                        use_container_width=True)
                
                # Prediction result
                predicted = result['predicted_class']
                max_confidence = max(result['confidence_scores'].values())
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2 style='margin: 0; color: white;'>Hasil Prediksi</h2>
                    <h1 style='margin: 20px 0; font-size: 48px;'>{predicted.upper()}</h1>
                    <p style='margin: 0; font-size: 16px; opacity: 0.9;'>
                        Confidence: {max_confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence scores
        if 'result' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Analisis Detail")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Confidence Score")
                fig_conf = plot_confidence_chart(result['confidence_scores'])
                st.pyplot(fig_conf)
                plt.close()
                
                # Display scores in metrics
                st.markdown("### Skor per Kelas")
                cols = st.columns(3)
                for i, (label, score) in enumerate(result['confidence_scores'].items()):
                    with cols[i]:
                        emoji = {'belum matang': '🔴', 'matang': '🟠', 'kematangan': '🟢'}
                        st.metric(
                            label=f"{emoji.get(label, '⚪')} {label.title()}",
                            value=f"{score:.1f}%"
                        )
            
            with col2:
                st.markdown("### Top 10 Fitur Terekstrak")
                fig_feat = plot_feature_importance(result['features'], top_n=10)
                st.pyplot(fig_feat)
                plt.close()
            
            # Feature table
            with st.expander("🔬 Lihat Semua Fitur (Advanced)"):
                df_features = pd.DataFrame([result['features']]).T
                df_features.columns = ['Nilai']
                df_features.index.name = 'Nama Fitur'
                st.dataframe(df_features, use_container_width=True)
            
            # Reset button
            if st.button("🔄 Reset & Upload Gambar Baru", key="reset_btn"):
                # Clear session state
                if 'result' in st.session_state:
                    del st.session_state['result']
                if 'original_image' in st.session_state:
                    del st.session_state['original_image']
                st.rerun()
    
    with tab2:
        st.header("Prediksi Batch (Multiple Images)")
        st.info("🚧 Fitur ini akan segera hadir! Untuk sementara gunakan mode Single Prediction.")
        
        # Placeholder untuk future development
        uploaded_files = st.file_uploader(
            "Upload multiple gambar",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload beberapa gambar sekaligus untuk prediksi batch",
            disabled=True
        )
    
    with tab3:
        st.header("📖 Tentang Sistem")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Tujuan Sistem
            Sistem ini dirancang untuk mengklasifikasikan tingkat kematangan cabai 
            secara otomatis menggunakan teknologi computer vision dan machine learning.
            
            ### 🔬 Metodologi
            
            **1. Segmentasi Background**
            - Otsu Thresholding
            - Morphological Operations
            - Contour Detection
            
            **2. Ekstraksi Fitur**
            - **HSV Color Features**: Mean, Std, Median, Min, Max untuk setiap channel (H, S, V)
            - **GLCM Texture Features**: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM
            
            **3. Klasifikasi**
            - Algorithm: Support Vector Machine (SVM)
            - Kernel: RBF (Radial Basis Function)
            - Feature Scaling: StandardScaler
            """)
        
        with col2:
            st.markdown(f"""
            ### 📊 Performa Model
            
            **Akurasi Overall**: {predictor.accuracy*100:.2f}%
            
            ### 🏷️ Kelas Prediksi
            
            1. **🔴 Belum Matang**
               - Cabai masih hijau
               - Tekstur keras
               - Belum siap panen
            
            2. **🟠 Matang**
               - Warna mulai berubah
               - Tekstur sedang
               - Siap untuk dipanen
            
            3. **🟢 Kematangan**
               - Warna merah cerah penuh
               - Tekstur optimal
               - Siap konsumsi/jual
            
            ### 👨‍💻 Developer
            Sistem dikembangkan menggunakan:
            - Python
            - OpenCV
            - Scikit-learn
            - Streamlit
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 💡 Tips Penggunaan
        
        - ✅ Gunakan gambar dengan pencahayaan yang baik
        - ✅ Pastikan cabai terlihat jelas
        - ✅ Background kontras lebih baik untuk segmentasi
        - ✅ Ukuran gambar tidak terlalu kecil (minimal 300x300px)
        - ⚠️ Hindari gambar blur atau terlalu gelap
        """)

if __name__ == "__main__":
    main()