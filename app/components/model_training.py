import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from components.feature_extraction import extract_features

# Fungsi untuk memuat gambar dari folder
def load_images_from_folder(folder):
    if not os.path.exists(folder):
        st.error(f"Folder {folder} tidak ditemukan!")
        return None, None

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    st.write(f"Jumlah file gambar yang ditemukan: {len(image_files)}")

    X = []  # Data gambar
    y = []  # Label gambar

    for filename in image_files:
        image_path = os.path.join(folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                img = np.array(Image.open(image_path).convert('L'))
            except Exception as e:
                st.write(f"Gambar tidak dapat dimuat: {filename}, Error: {e}")
                continue

        # Resize gambar menjadi 128x128
        img_resized = cv2.resize(img, (128, 128))
        X.append(img_resized)

        # Ambil label dari nama file (misalnya, angka sebelum underscore "_")
        try:
            label = int(filename.split('_')[0])
            y.append(label)
        except ValueError:
            st.write(f"Format nama file tidak sesuai: {filename}")
            continue

    return np.array(X), np.array(y)

# Streamlit UI
st.title("Aplikasi Pengenalan Pola")
st.header("1. Pilih Folder Data")

# Input path folder
folder_path = st.text_input("Masukkan path folder data:")

# Inisialisasi session state untuk menyimpan data
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.X_transformed = None  # Untuk menyimpan hasil transformasi

if st.button("Muat Data"):
    if folder_path:
        X, y = load_images_from_folder(folder_path)
        if X is not None and y is not None:
            st.session_state.data_loaded = True
            st.session_state.X = X
            st.session_state.y = y
            st.write(f"Jumlah gambar yang diproses: {len(X)}")
            st.write(f"Jumlah label yang diproses: {len(y)}")
        else:
            st.error("Data tidak berhasil dimuat!")
    else:
        st.error("Harap masukkan path folder data.")

# Langkah kedua: Pilih apakah ingin menggunakan fitur ekstraksi atau langsung tanpa fitur
if st.session_state.data_loaded:
    st.header("2. Pilih Opsi Fitur Ekstraksi")
    
    # Checkbox untuk menggunakan ekstraksi fitur
    use_feature_extraction = st.checkbox("Gunakan fitur ekstraksi?")
    
    # Checkbox untuk langsung tanpa ekstraksi fitur
    skip_extraction = st.checkbox("Tanpa fitur ekstraksi (lanjut ke langkah 3)")

    if use_feature_extraction and not skip_extraction:
        # Pilihan metode ekstraksi
        extraction_method = st.selectbox("Pilih metode ekstraksi fitur:", ["PCA", "HOG"])

        if st.button("Lakukan Ekstraksi Fitur"):
            # Ekstraksi fitur berdasarkan metode yang dipilih
            try:
                X_transformed = extract_features(st.session_state.X, extraction_method)
                st.session_state.X_transformed = X_transformed
                st.session_state.features_extracted = True
                st.write(f"Dimensi fitur setelah {extraction_method}: {X_transformed.shape}")
            except ValueError as e:
                st.error(f"Terjadi kesalahan: {e}")
    
    if skip_extraction and not use_feature_extraction:
        # Langsung lanjut tanpa ekstraksi fitur
        st.session_state.X_transformed = np.array([img.flatten() for img in st.session_state.X])
        st.session_state.features_extracted = False
        st.write(f"Fitur asli digunakan tanpa ekstraksi. Dimensi: {st.session_state.X_transformed.shape}")

    # Langkah ketiga: Pembagian dataset
    if (skip_extraction or st.session_state.features_extracted):
        st.header("3. Pembagian Dataset")

        # Pilihan persentase data uji
        test_size_option = st.selectbox(
            "Pilih persentase data uji:",
            options=["50%", "40%", "30%", "20%"],
            index=0  # Default 50%
        )

        # Konversi pilihan ke nilai float untuk test_size
        test_size_map = {"50%": 0.5, "40%": 0.4, "30%": 0.3, "20%": 0.2}
        test_size = test_size_map[test_size_option]

        if st.button("Bagi Dataset"):
            # Membagi data menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X_transformed, st.session_state.y, test_size=test_size, random_state=42
            )
            # Normalisasi data menggunakan StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Menyimpan hasil pembagian di session state
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.write(f"Persentase data uji: {test_size_option}")
            st.write(f"Jumlah data latih: {len(X_train)}, Jumlah data uji: {len(X_test)}")
