import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

def load_data_section():
    st.header("1. Pilih Folder Data")
    folder_path = st.text_input("Masukkan path folder dataset:", "")
    
    if st.button("Muat Data"):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            st.write(f"Jumlah file gambar yang ditemukan: {len(image_files)}")
            
            X, y = [], []
            for filename in image_files:
                image_path = os.path.join(folder_path, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    try:
                        img = np.array(Image.open(image_path).convert('L'))
                    except Exception as e:
                        st.write(f"Gagal memuat file {filename}: {e}")
                        continue

                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)

                try:
                    label = int(filename.split('_')[0])
                    y.append(label)
                except ValueError:
                    st.write(f"Format nama file tidak sesuai: {filename}")
                    continue

            return np.array(X), np.array(y)
        else:
            st.error("Path folder tidak valid!")
            return None, None
    return None, None
