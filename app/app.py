import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from components.feature_extraction import extract_features  # Pastikan fungsi ini menangani HOG dan PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
import pickle
from io import BytesIO

# Konfigurasi Halaman (Harus dijalankan pertama kali sebelum fungsi streamlit lainnya)
st.set_page_config(
    page_title="Pengenalan Digit Tulisan Tangan",
    page_icon="✍️",
    layout="centered"
)

import tempfile
import hashlib

# Gunakan folder temp di Drive C (yang memiliki banyak ruang kosong) untuk menghindari error disk full di Drive D
project_hash = hashlib.md5(os.path.abspath(__file__).encode('utf-8')).hexdigest()
STATE_FILE = os.path.join(tempfile.gettempdir(), f"streamlit_digit_recognizer_{project_hash}.pkl")
KEYS_TO_PERSIST = [
    "current_step",
    "data_loaded",
    "X",
    "y",
    "X_transformed",
    "scalers",
    "models",
    "k",
    "n_estimators",
    "n_components",
    "use_feature_extraction",
    "skip_extraction",
    "extraction_method",
    "pca_transformer",
    "X_train_scaled",
    "X_test_scaled",
    "y_train",
    "y_test",
    "test_size_option_str",
    "algorithm_option_str",
    "test_extraction_option_str"
]

def save_state():
    try:
        state_dict = {}
        for key in KEYS_TO_PERSIST:
            if key in st.session_state:
                state_dict[key] = st.session_state[key]
        with open(STATE_FILE, "wb") as f:
            pickle.dump(state_dict, f)
    except Exception as e:
        import traceback
        print(f"SAVE STATE ERROR: {e}")
        traceback.print_exc()

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                state_dict = pickle.load(f)
            for key, val in state_dict.items():
                st.session_state[key] = val
        except Exception as e:
            import traceback
            print(f"LOAD STATE ERROR: {e}")
            traceback.print_exc()

# load state dari penyimpanan lokal jika baru pertama kali dijalankan
if "state_loaded" not in st.session_state:
    load_state()
    st.session_state.state_loaded = True

# Custom CSS untuk UI/UX Premium, Modern, dan Clean
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main Page Style */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 50% 30%, #171725 0%, #07070a 100%) !important;
    font-family: 'Outfit', sans-serif !important;
    color: #f1f5f9 !important;
}

/* Hide Default Streamlit Header */
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0) !important;
}

/* Glassmorphism Card Style (Sangat Halus & Modern) */
.glass-card {
    background: rgba(20, 20, 35, 0.6) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 40px !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 30px !important;
}

/* Typography Overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
}

/* Steps Progress Bar Styling */
.steps-container {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 45px !important;
    position: relative !important;
    padding: 0 10px !important;
    width: 100% !important;
}

.steps-line {
    position: absolute !important;
    top: 20px !important;
    left: 0 !important;
    right: 0 !important;
    height: 4px !important;
    background: rgba(255, 255, 255, 0.04) !important;
    z-index: 1 !important;
}

.steps-line-progress {
    position: absolute !important;
    top: 20px !important;
    left: 0 !important;
    height: 4px !important;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%) !important;
    z-index: 2 !important;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.step-item {
    position: relative !important;
    z-index: 3 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 90px !important;
}

.step-circle {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    background: #12121e !important;
    border: 2px solid rgba(255, 255, 255, 0.08) !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    font-weight: 600 !important;
    color: #475569 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
}

.step-item.active .step-circle {
    background: #0f172a !important;
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
}

.step-item.completed .step-circle {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    border-color: transparent !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(129, 140, 248, 0.35) !important;
}

.step-label {
    margin-top: 10px !important;
    font-size: 0.75rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
    text-align: center !important;
    transition: color 0.3s ease !important;
    white-space: nowrap !important;
}

.step-item.active .step-label {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

.step-item.completed .step-label {
    color: #818cf8 !important;
}

/* Customize Streamlit Buttons (Melayang, Gradient, Glowing) */
div.stButton > button {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(129, 140, 248, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100%;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(129, 140, 248, 0.4) !important;
    color: #ffffff !important;
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Secondary Button style for Navigation */
.nav-back-btn div.stButton > button {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
}

.nav-back-btn div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}

/* Disabled/Aria-Disabled Button Styling (Rapi & Menyatu) */
div.stButton > button:disabled, 
div.stButton > button[disabled],
div.stButton > button[aria-disabled="true"] {
    background: rgba(255, 255, 255, 0.02) !important;
    color: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* Reset Button Style (Elegan, Pojok Kanan Atas) */
.reset-btn-container div.stButton > button {
    background: transparent !important;
    color: #ef4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.35) !important;
    box-shadow: none !important;
    width: auto !important;
    margin: 0 0 0 auto !important;
    display: block !important;
    padding: 8px 18px !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
}

.reset-btn-container div.stButton > button:hover {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: rgba(239, 68, 68, 0.55) !important;
    color: #fca5a5 !important;
    transform: translateY(-1px) !important;
}

/* Custom File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 2px dashed rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 25px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.02) !important;
}

/* Alert Notification Style */
.custom-alert {
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 0.95rem;
}
.custom-alert-success {
    background: rgba(16, 185, 129, 0.12) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
    color: #34d399 !important;
}
.custom-alert-info {
    background: rgba(56, 189, 248, 0.12) !important;
    border-color: rgba(56, 189, 248, 0.25) !important;
    color: #38bdf8 !important;
}

/* Prediction Result Card */
.prediction-card {
    background: rgba(129, 140, 248, 0.1) !important;
    border: 2px solid rgba(129, 140, 248, 0.3) !important;
    box-shadow: 0 0 30px rgba(129, 140, 248, 0.2) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    text-align: center;
    margin-top: 10px;
}

/* Form inputs styling */
input[type="text"] {
    background-color: rgba(10, 10, 15, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Fungsi untuk memuat gambar dari folder
def load_images_from_folder(folder):
    if not os.path.exists(folder):
        st.error(f"Folder {folder} tidak ditemukan!")
        return None, None

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    if len(image_files) == 0:
        st.warning("Tidak ditemukan file gambar .png atau .jpg di folder ini!")
        return None, None

    st.info(f"Jumlah file gambar yang ditemukan: {len(image_files)}")

    X = []  # Data gambar
    y = []  # Label gambar

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for idx, filename in enumerate(image_files):
        # Update progress bar
        progress_bar.progress((idx + 1) / len(image_files))
        if idx % 50 == 0 or idx == len(image_files) - 1:
            status_text.text(f"Memuat berkas {idx + 1}/{len(image_files)}: {filename}")

        image_path = os.path.join(folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                img = np.array(Image.open(image_path).convert('L'))
            except Exception as e:
                continue

        # Resize gambar menjadi 128x128
        img_resized = cv2.resize(img, (128, 128))
        X.append(img_resized)

        # Ambil label dari nama file (misalnya, angka sebelum underscore "_")
        try:
            label = int(filename.split('_')[0])
            y.append(label)
        except ValueError:
            continue

    # Hapus progress loader setelah selesai
    progress_bar.empty()
    status_text.empty()

    return np.array(X), np.array(y)

# Fungsi untuk memprediksi label untuk gambar baru
def predict_image(model, scaler, image_features):
    # Normalisasi fitur
    features_scaled = scaler.transform([image_features])
    # Prediksi
    predicted_label = model.predict(features_scaled)
    return predicted_label[0]

# Fungsi untuk mengonversi gambar ke base64
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Inisialisasi session state secara independen dan aman (mencegah bug reload Streamlit)
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_transformed" not in st.session_state:
    st.session_state.X_transformed = {}
if "scalers" not in st.session_state:
    st.session_state.scalers = {}
if "models" not in st.session_state:
    st.session_state.models = {}
if "k" not in st.session_state:
    st.session_state.k = 3
if "n_estimators" not in st.session_state:
    st.session_state.n_estimators = 100
if "n_components" not in st.session_state:
    st.session_state.n_components = 50
if "use_feature_extraction" not in st.session_state:
    st.session_state.use_feature_extraction = False
if "skip_extraction" not in st.session_state:
    st.session_state.skip_extraction = False
if "pca_transformer" not in st.session_state:
    st.session_state.pca_transformer = None
if "test_size_option_str" not in st.session_state:
    st.session_state.test_size_option_str = "40%"
if "algorithm_option_str" not in st.session_state:
    st.session_state.algorithm_option_str = "SVM (Support Vector Machine)"
if "test_extraction_option_str" not in st.session_state:
    st.session_state.test_extraction_option_str = ""

# Render Judul Web & Reset Button (Header Row Symmetrical)
col_title, col_reset = st.columns([3, 1])
with col_title:
    st.markdown("<h2 style='margin: 0; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 2.3rem; letter-spacing: -1px;'>Pengenalan Digit</h2>", unsafe_allow_html=True)
    st.markdown("<p style='margin: 0 0 25px 0; color: #64748b; font-size: 0.95rem; font-weight: 400;'>Sistem Klasifikasi Digit Tangan dengan Machine Learning</p>", unsafe_allow_html=True)
with col_reset:
    st.markdown("<div class='reset-btn-container' style='display: flex; justify-content: flex-end; align-items: center; margin-top: 5px;'>", unsafe_allow_html=True)
    if st.button("Reset Proyek", key="reset_app_btn", help="Hapus seluruh cache data & model untuk mulai dari awal"):
        if os.path.exists(STATE_FILE):
            try:
                os.remove(STATE_FILE)
            except:
                pass
        # Reset seluruh session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Render Progress Bar di bagian atas
def render_step_progress_bar(current_step):
    steps = [
        "Muat Data",
        "Ekstraksi",
        "Bagi Data",
        "Latih Model",
        "Uji Model"
    ]
    progress_percentage = (current_step - 1) / (len(steps) - 1) * 100
    
    step_items_html = ""
    for i, name in enumerate(steps):
        step_num = i + 1
        status_class = ""
        if step_num < current_step:
            status_class = "completed"
        elif step_num == current_step:
            status_class = "active"
            
        step_items_html += f'<div class="step-item {status_class}"><div class="step-circle">{step_num}</div><div class="step-label">{name}</div></div>'
        
    progress_html = f'<div class="steps-container"><div class="steps-line"></div><div class="steps-line-progress" style="width: {progress_percentage}%;"></div>{step_items_html}</div>'
    st.html(progress_html)

render_step_progress_bar(st.session_state.current_step)

# --- ALUR WIZARD PER LANGKAH ---

# LANGKAH 1: PEMUATAN DATA
if st.session_state.current_step == 1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("1. Pilih & Muat Dataset")
    st.write("Pilih metode pemuatan dataset yang ingin Anda gunakan:")
    
    tab_path, tab_upload = st.tabs(["📁 Input Path Lokal", "📤 Seret & Lepas File"])
    
    with tab_path:
        st.write("Masukkan path absolut folder yang berisikan gambar dataset digit tulisan tangan Anda.")
        default_path = r"D:\deteksi angka\app\DS-2"
        folder_path = st.text_input("Masukkan path folder data:", value=default_path, key="folder_path_input")
        
        st.write("") # Spacing
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Muat Data (Path)", use_container_width=True, key="load_data_path_btn"):
                if folder_path:
                    with st.spinner("Memproses seluruh berkas citra dataset..."):
                        X, y = load_images_from_folder(folder_path)
                    if X is not None and y is not None and len(X) > 0:
                        st.session_state.data_loaded = True
                        st.session_state.X = X
                        st.session_state.y = y
                        st.html(f'<div class="custom-alert custom-alert-success"><strong>Pemuatan Berhasil!</strong><br>• Jumlah gambar diproses: <strong>{len(X)}</strong><br>• Jumlah label diidentifikasi: <strong>{len(y)}</strong></div>')
                        save_state()
                    else:
                        st.error("Data tidak berhasil dimuat! Pastikan path folder benar dan terdapat file gambar format .png/.jpg di dalamnya.")
                else:
                    st.error("Harap masukkan path folder data.")
                    
    with tab_upload:
        st.write("Unggah atau seret gambar dataset digit Anda ke area di bawah.")
        uploaded_files = st.file_uploader(
            "Pilih atau seret gambar (.png, .jpg, .jpeg) di sini:",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="dataset_file_uploader"
        )
        
        st.write("") # Spacing
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Muat Data (Unggah)", use_container_width=True, key="load_data_upload_btn"):
                if uploaded_files:
                    X = []
                    y = []
                    
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        if idx % 50 == 0 or idx == len(uploaded_files) - 1:
                            status_text.text(f"Memproses berkas {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                            
                        try:
                            # Membaca berkas gambar dari buffer memori
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                            
                            if img is not None:
                                img_resized = cv2.resize(img, (128, 128))
                                X.append(img_resized)
                                
                                # Ambil label dari nama file (misalnya, angka sebelum underscore "_")
                                try:
                                    label = int(uploaded_file.name.split('_')[0])
                                    y.append(label)
                                except ValueError:
                                    continue
                        except Exception as e:
                            continue
                            
                    progress_bar.empty()
                    status_text.empty()
                    
                    if len(X) > 0:
                        st.session_state.data_loaded = True
                        st.session_state.X = np.array(X)
                        st.session_state.y = np.array(y)
                        st.html(f'<div class="custom-alert custom-alert-success"><strong>Pemuatan Berhasil!</strong><br>• Jumlah gambar diproses: <strong>{len(X)}</strong><br>• Jumlah label diidentifikasi: <strong>{len(y)}</strong></div>')
                        save_state()
                    else:
                        st.error("Tidak ada gambar valid yang berhasil diproses!")
                else:
                    st.error("Harap pilih atau seret berkas gambar terlebih dahulu.")
            
    if st.session_state.data_loaded:
        st.html(f'<div class="custom-alert custom-alert-info"><strong>Status:</strong> Dataset aktif saat ini berisi {len(st.session_state.X)} sampel. Klik tombol \'Selanjutnya\' di bawah untuk mengekstrak fitur.</div>')
    st.markdown("</div>", unsafe_allow_html=True)

# LANGKAH 2: PILIH FITUR EKSTRAKSI
elif st.session_state.current_step == 2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("2. Pilihan Opsi Fitur Ekstraksi")
    st.write("Pilih bagaimana Anda ingin mengekstrak ciri unik (fitur) dari piksel gambar sebelum dilatih.")
    
    use_feature_extraction = st.checkbox("Gunakan fitur ekstraksi?", value=st.session_state.use_feature_extraction)
    st.session_state.use_feature_extraction = use_feature_extraction

    skip_extraction = st.checkbox("Tanpa fitur ekstraksi (Flatten Pixel Langsung)", value=st.session_state.skip_extraction)
    st.session_state.skip_extraction = skip_extraction

    if use_feature_extraction and skip_extraction:
        st.error("Hanya boleh memilih salah satu opsi!")
    elif use_feature_extraction:
        # Menentukan index pilihan default agar tersimpan saat browser restart
        saved_method = st.session_state.get('extraction_method', 'HOG')
        default_method_idx = 0 if saved_method == 'HOG' else 1
        
        extraction_method = st.selectbox(
            "Pilih metode ekstraksi fitur:", 
            ["HOG", "PCA"],
            index=default_method_idx
        )
        st.session_state.extraction_method = extraction_method

        if extraction_method == "PCA":
            st.write(f"Jumlah komponen PCA telah ditetapkan ke **{st.session_state.n_components}**.")

        st.write("") # Spacing
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Lakukan Ekstraksi Fitur", use_container_width=True):
                with st.spinner(f"Mengekstrak ciri gambar menggunakan {extraction_method}..."):
                    try:
                        if extraction_method == "PCA":
                            # Melakukan PCA lokal untuk menyimpan objek PCA hasil fit ke session state
                            from sklearn.decomposition import PCA as sklearn_PCA
                            X_flattened = np.array([img.flatten() for img in st.session_state.X])
                            pca = sklearn_PCA(n_components=st.session_state.n_components, random_state=42)
                            X_transformed = pca.fit_transform(X_flattened)
                            st.session_state.pca_transformer = pca
                        else:
                            X_transformed = extract_features(st.session_state.X, method=extraction_method, n_components=st.session_state.n_components)
                        
                        if X_transformed is not None:
                            st.session_state.X_transformed[extraction_method] = X_transformed
                            st.session_state.scalers[extraction_method] = StandardScaler()
                            st.html(f'<div class="custom-alert custom-alert-success"><strong>Ekstraksi {extraction_method} Sukses!</strong><br>Dimensi matriks fitur baru: <strong>{X_transformed.shape}</strong></div>')
                    except ValueError as e:
                        st.error(f"Terjadi kesalahan: {e}")
                    
    elif skip_extraction:
        st.write("") # Spacing
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Lakukan Flatten Gambar", use_container_width=True):
                with st.spinner("Meratakan matriks gambar menjadi vektor 1D..."):
                    try:
                        X_flatten = np.array([img.flatten() for img in st.session_state.X])
                        st.session_state.X_transformed["None"] = X_flatten
                        st.session_state.scalers["None"] = StandardScaler()
                        st.html(f'<div class="custom-alert custom-alert-success"><strong>Flatten Berhasil!</strong><br>Dimensi matriks fitur baru: <strong>{X_flatten.shape}</strong></div>')
                    except ValueError as e:
                        st.error(f"Terjadi kesalahan: {e}")

    # Indikasi metode apa saja yang telah berhasil diekstraksi
    if len(st.session_state.X_transformed) > 0:
        methods_str = ", ".join([f"<strong>{m}</strong>" for m in st.session_state.X_transformed.keys()])
        st.html(f'<div class="custom-alert custom-alert-info"><strong>Metode Siap Pakai:</strong> {methods_str}. Anda bisa lanjut ke langkah pembagian dataset.</div>')
    st.markdown("</div>", unsafe_allow_html=True)

# LANGKAH 3: PEMBAGIAN DATASET
elif st.session_state.current_step == 3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("3. Pembagian Dataset (Train/Test Split)")
    st.write("Bagi data Anda menjadi kumpulan data latih untuk pelatihan model dan data uji untuk pengujian akurasi.")
    
    if len(st.session_state.X_transformed) == 0:
        st.error("Metode ekstraksi belum siap. Kembali ke langkah sebelumnya untuk mengekstrak fitur.")
    else:
        methods_str = ", ".join([f"<strong>{m}</strong>" for m in st.session_state.X_transformed.keys()])
        st.markdown(f"Fitur yang telah siap dibagi: {methods_str}", unsafe_allow_html=True)
        
        # Opsi persentase data uji dengan index dinamis
        saved_test_size = st.session_state.get('test_size_option_str', "40%")
        default_test_idx = ["40%", "30%", "20%"].index(saved_test_size) if saved_test_size in ["40%", "30%", "20%"] else 0
        
        test_size_option = st.selectbox(
            "Pilih persentase data uji:",
            options=["40%", "30%", "20%"],
            index=default_test_idx
        )
        st.session_state.test_size_option_str = test_size_option
        test_size_map = {"40%": 0.4, "30%": 0.3, "20%": 0.2}
        test_size = test_size_map[test_size_option]

        st.write("") # Spacing
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Bagi Dataset", use_container_width=True):
                try:
                    # Inisialisasi ulang
                    st.session_state.X_train_scaled = {}
                    st.session_state.X_test_scaled = {}
                    st.session_state.y_train = {}
                    st.session_state.y_test = {}

                    for method, X in st.session_state.X_transformed.items():
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, st.session_state.y, test_size=test_size, random_state=42
                        )

                        # Normalisasi data
                        scaler = st.session_state.scalers[method]
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        st.session_state.X_train_scaled[method] = X_train_scaled
                        st.session_state.X_test_scaled[method] = X_test_scaled
                        st.session_state.y_train[method] = y_train
                        st.session_state.y_test[method] = y_test

                        st.html(f'<div style="border-left: 3px solid #818cf8; padding-left: 15px; margin-bottom: 15px;"><strong>Metode: {method}</strong><br>• Jumlah data latih (training): <strong>{len(X_train)}</strong> sampel<br>• Jumlah data uji (testing): <strong>{len(X_test)}</strong> sampel</div>')
                    
                    st.success("Pembagian data dan penskalaan standar (StandardScaler) berhasil!")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membagi dataset: {e}")
                    
        if "X_train_scaled" in st.session_state and len(st.session_state.X_train_scaled) > 0:
            st.html(f'<div class="custom-alert custom-alert-info"><strong>Status:</strong> Dataset train/test split siap dengan ukuran uji <strong>{test_size_option}</strong>.</div>')
    st.markdown("</div>", unsafe_allow_html=True)

# LANGKAH 4: PELATIHAN MODEL & EVALUASI
elif st.session_state.current_step == 4:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("4. Pemilihan Algoritma & Pelatihan Model")
    st.write("Latih model Machine Learning menggunakan data latih dan evaluasi hasilnya langsung dengan metrik visual.")
    
    # Pilih algoritma dengan index dinamis
    saved_algo = st.session_state.get('algorithm_option_str', "SVM (Support Vector Machine)")
    algo_list = ["SVM (Support Vector Machine)", "KNN (K-Nearest Neighbors)", "Random Forest"]
    default_algo_idx = algo_list.index(saved_algo) if saved_algo in algo_list else 0

    algorithm_option = st.selectbox(
        "Pilih algoritma pembelajaran mesin:",
        options=algo_list,
        index=default_algo_idx
    )
    st.session_state.algorithm_option_str = algorithm_option

    # UI Parameter Dinamis
    if algorithm_option == "KNN (K-Nearest Neighbors)":
        k = st.slider("Pilih jumlah tetangga (k):", min_value=1, max_value=20, value=st.session_state.k, step=1)
        st.session_state.k = k
    elif algorithm_option == "Random Forest":
        n_estimators = st.slider("Pilih jumlah pohon (n_estimators):", min_value=10, max_value=200, value=st.session_state.n_estimators, step=10)
        st.session_state.n_estimators = n_estimators

    # Fungsi internal untuk training & evaluasi
    def train_and_evaluate_model(method, X_train, X_test, y_train, y_test, model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.session_state.models[method] = model  # Simpan model

        st.markdown(f"#### Hasil Analisis Metode: **{method}**")

        if y_pred is not None:
            accuracy = accuracy_score(y_test, y_pred)
            st.html(f'<div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); padding: 15px; border-radius: 10px; margin-bottom: 20px;">Akurasi Prediksi ({method}): <strong style="font-size: 1.4rem; color: #38bdf8;">{accuracy * 100:.2f}%</strong></div>')

            # Laporan Klasifikasi
            st.write("**Laporan Klasifikasi:**")
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(2)
            
            st.table(
                report_df.style
                .highlight_max(axis=0, color='rgba(52, 211, 153, 0.25)')
                .format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"})
                .background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                .set_properties(**{'text-align': 'center'})
            )

            # Confusion Matrix dengan tema gelap
            st.write("**Confusion Matrix:**")
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted Label', color='#94a3b8', fontsize=10)
            ax.set_ylabel('True Label', color='#94a3b8', fontsize=10)
            ax.tick_params(colors='#94a3b8')
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.1))
            
            st.pyplot(fig)
            plt.close(fig)

    # Tombol Pelatihan sesuai algoritma
    st.write("") # Spacing
    col_l, col_btn, col_r = st.columns([1, 2, 1])
    
    with col_btn:
        if algorithm_option == "SVM (Support Vector Machine)":
            if st.button("Latih Model SVM", use_container_width=True):
                with st.spinner("Melatih model SVM di seluruh dataset aktif..."):
                    for method in st.session_state.X_train_scaled.keys():
                        model = SVC(kernel='linear', probability=True, random_state=42)
                        X_train = st.session_state.X_train_scaled[method]
                        X_test = st.session_state.X_test_scaled[method]
                        y_train = st.session_state.y_train[method]
                        y_test = st.session_state.y_test[method]
                        train_and_evaluate_model(method, X_train, X_test, y_train, y_test, model)

        elif algorithm_option == "KNN (K-Nearest Neighbors)":
            if st.button("Latih Model KNN", use_container_width=True):
                with st.spinner(f"Melatih model KNN (k={st.session_state.k})..."):
                    for method in st.session_state.X_train_scaled.keys():
                        model = KNeighborsClassifier(n_neighbors=st.session_state.k)
                        X_train = st.session_state.X_train_scaled[method]
                        X_test = st.session_state.X_test_scaled[method]
                        y_train = st.session_state.y_train[method]
                        y_test = st.session_state.y_test[method]
                        train_and_evaluate_model(method, X_train, X_test, y_train, y_test, model)

        elif algorithm_option == "Random Forest":
            if st.button("Latih Model Random Forest", use_container_width=True):
                with st.spinner(f"Melatih model Random Forest ({st.session_state.n_estimators} trees)..."):
                    for method in st.session_state.X_train_scaled.keys():
                        model = RandomForestClassifier(n_estimators=st.session_state.n_estimators, random_state=42)
                        X_train = st.session_state.X_train_scaled[method]
                        X_test = st.session_state.X_test_scaled[method]
                        y_train = st.session_state.y_train[method]
                        y_test = st.session_state.y_test[method]
                        train_and_evaluate_model(method, X_train, X_test, y_train, y_test, model)

    if len(st.session_state.models) > 0:
        st.html('<div class="custom-alert custom-alert-info"><strong>Model Terlatih Aktif!</strong> Lanjut ke halaman terakhir untuk melakukan pengujian prediksi visual.</div>')
    st.markdown("</div>", unsafe_allow_html=True)

# LANGKAH 5: PENGUJIAN PREDIKSI FOTO
elif st.session_state.current_step == 5:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("5. Uji Model dengan Foto Baru")
    st.write("Unggah foto tulisan tangan Anda untuk memprediksi angka secara langsung berdasarkan model terlatih.")

    trained_methods = list(st.session_state.models.keys())
    if len(trained_methods) == 0:
        st.error("Belum ada model terlatih. Kembali ke langkah ke-4 terlebih dahulu untuk melatih model.")
    else:
        # Pilihan metode ekstraksi dengan index dinamis
        saved_test_opt = st.session_state.get('test_extraction_option_str', trained_methods[0])
        default_test_idx = trained_methods.index(saved_test_opt) if saved_test_opt in trained_methods else 0
        
        st.write("Pilih Metode Ekstraksi Fitur untuk Pengujian:")
        test_extraction_option = st.selectbox(
            "Metode Ekstraksi:",
            options=trained_methods,
            index=default_test_idx
        )
        st.session_state.test_extraction_option_str = test_extraction_option

        # File uploader
        uploaded_file = st.file_uploader("Pilih file gambar baru:", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                # Membaca gambar asli
                original_image = Image.open(uploaded_file).convert('L')  # Convert ke grayscale
                original_image_resized = original_image.resize((128, 128))
                original_image_array = np.array(original_image_resized)
            except Exception as e:
                st.error(f"Gagal memproses foto yang diunggah: {e}")
                original_image_array = None

            if original_image_array is not None:
                # Preprocessing foto
                try:
                    img_preprocessed = cv2.resize(original_image_array, (128, 128))
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat meresize foto: {e}")
                    img_preprocessed = None

                if img_preprocessed is not None:
                    # Ekstraksi fitur berdasarkan pilihan pengguna
                    try:
                        if test_extraction_option == "HOG":
                            if "HOG" in st.session_state.X_transformed:
                                img_features = extract_features([img_preprocessed], method="HOG")
                                img_features = img_features[0]  # extract_features mengembalikan array
                            else:
                                st.error("Ekstraksi fitur HOG belum dilakukan selama pelatihan.")
                                img_features = None
                        elif test_extraction_option == "PCA":
                            if st.session_state.pca_transformer is not None:
                                img_flattened = img_preprocessed.flatten()
                                # Memakai objek transformer yang sudah dilatih (bukan mem-fit ulang) untuk mencegah ValueError
                                img_features = st.session_state.pca_transformer.transform([img_flattened])[0]
                            else:
                                st.error("Model transformer PCA belum dilatih.")
                                img_features = None
                        elif test_extraction_option == "None":
                            img_features = img_preprocessed.flatten()
                        else:
                            st.error("Metode ekstraksi fitur tidak dikenali.")
                            img_features = None
                    except ValueError as e:
                        st.error(f"Terjadi kesalahan saat ekstraksi fitur: {e}")
                        img_features = None

                    if img_features is not None:
                        # Normalisasi data menggunakan scaler yang sesuai
                        scaler_key = test_extraction_option if test_extraction_option in st.session_state.scalers else "None"
                        scaler = st.session_state.scalers.get(scaler_key, None)

                        if scaler is not None:
                            try:
                                img_features_scaled = scaler.transform([img_features])
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat normalisasi fitur: {e}")
                                img_features_scaled = None
                        else:
                            st.error(f"Scaler untuk metode {test_extraction_option} tidak ditemukan.")
                            img_features_scaled = None

                        if img_features_scaled is not None:
                            # Prediksi menggunakan model yang sesuai
                            model = st.session_state.models.get(test_extraction_option, None)

                            if model is not None:
                                try:
                                    prediction = predict_image(model, scaler, img_features)

                                    # Mengonversi img_preprocessed kembali ke format PIL untuk ditampilkan
                                    processed_image = Image.fromarray(img_preprocessed)

                                    # Mengonversi gambar ke base64
                                    processed_image_base64 = get_image_base64(processed_image)

                                    # Menampilkan layout prediksi berdampingan
                                    col_img, col_pred = st.columns([1, 1])
                                    with col_img:
                                        st.write("**Gambar Masukan (Preprocessing):**")
                                        st.html(f'<div style="text-align: center; background: rgba(0,0,0,0.2); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);"><img src="data:image/png;base64,{processed_image_base64}" width="180" style="border-radius: 8px; box-shadow: 0 8px 16px rgba(0,0,0,0.4);" /></div>')
                                    with col_pred:
                                        st.write("**Hasil Tebakan Model:**")
                                        st.html(f'<div class="prediction-card"><div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px;">Prediksi Digit</div><div style="font-size: 5.5rem; font-weight: 700; color: #ffffff; text-shadow: 0 0 25px rgba(56, 189, 248, 0.75); line-height: 1;">{prediction}</div><div style="font-size: 0.9rem; color: #38bdf8; font-weight: 600; margin-top: 15px;">Model: {model.__class__.__name__} ({test_extraction_option})</div></div>')
                                except Exception as e:
                                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
                            else:
                                st.error(f"Model untuk metode {test_extraction_option} tidak tersedia.")
    st.markdown("</div>", unsafe_allow_html=True)


# --- AREA TOMBOL NAVIGASI DI BAWAH ---
st.write("")
col_prev, col_spacer, col_next = st.columns([2, 3, 2])

# Tombol Kembali
with col_prev:
    if st.session_state.current_step > 1:
        st.markdown("<div class='nav-back-btn'>", unsafe_allow_html=True)
        if st.button("← Kembali", key="wizard_prev_btn", use_container_width=True):
            st.session_state.current_step -= 1
            save_state()  # Simpan state ke cache
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# Tombol Selanjutnya dengan validasi data
with col_next:
    if st.session_state.current_step < 5:
        can_go_next = False
        warning_msg = ""
        
        # Validasi langkah sebelum lanjut
        if st.session_state.current_step == 1:
            can_go_next = st.session_state.data_loaded
            warning_msg = "Anda wajib memuat dataset terlebih dahulu sebelum melanjutkan!"
        elif st.session_state.current_step == 2:
            can_go_next = len(st.session_state.X_transformed) > 0
            warning_msg = "Anda wajib memilih dan melakukan ekstraksi fitur atau flatten gambar terlebih dahulu!"
        elif st.session_state.current_step == 3:
            can_go_next = "X_train_scaled" in st.session_state and len(st.session_state.X_train_scaled) > 0
            warning_msg = "Anda wajib menekan tombol Bagi Dataset terlebih dahulu!"
        elif st.session_state.current_step == 4:
            can_go_next = len(st.session_state.models) > 0
            warning_msg = "Anda wajib menekan tombol Latih Model terlebih dahulu!"

        if can_go_next:
            if st.button("Selanjutnya →", key="wizard_next_btn", use_container_width=True):
                st.session_state.current_step += 1
                save_state()  # Simpan state ke cache
                st.rerun()
        else:
            # Tombol dinonaktifkan dengan tooltip/help berisi peringatan
            st.button("Selanjutnya →", disabled=True, key="wizard_next_btn_disabled", help=warning_msg, use_container_width=True)

# Pemicu Auto-Save Akhir Berkas (Menjamin persistensi instan atas setiap input interaktif)
save_state()
