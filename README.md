# ✍️ Sistem Klasifikasi & Pengenalan Digit Tulisan Tangan

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8.svg)](https://opencv.org/)

Aplikasi web interaktif berbasis **Streamlit** untuk melatih, mengevaluasi, dan menguji model Machine Learning dalam mengenali angka tulisan tangan (0-9). Aplikasi ini dirancang dengan antarmuka pengguna (UI/UX) premium bertema **Dark Mode & Glassmorphism** serta tata letak alur pengerjaan 5 langkah wizard yang intuitif.

---

## 🌟 Fitur Utama

1. **Alur Wizard 5 Langkah Terbimbing**:
   * **Langkah 1 (Muat Data)**: Memuat dataset gambar digit tulisan tangan Anda.
   * **Langkah 2 (Ekstraksi)**: Ekstraksi fitur unik dari gambar piksel.
   * **Langkah 3 (Bagi Data)**: Memisahkan data menjadi subset latih (train) dan uji (test).
   * **Langkah 4 (Latih Model)**: Pelatihan model ML dan visualisasi metrik evaluasi.
   * **Langkah 5 (Uji Model)**: Uji prediksi real-time dengan mengunggah gambar baru.

2. **Dua Metode Pemuatan Dataset (Langkah 1)**:
   * 📁 **Input Path Lokal**: Membaca cepat ribuan file citra langsung dari direktori komputer Anda.
   * 📤 **Seret & Lepas File**: Pengguna dapat langsung menyeleksi atau men-drag-and-drop banyak file gambar (atau folder) langsung ke browser. File didekode secara in-memory untuk mendukung kemudahan cloud deployment.

3. **Metode Ekstraksi Fitur Variatif (Langkah 2)**:
   * **HOG (Histogram of Oriented Gradients)**: Mengidentifikasi distribusi orientasi gradien tepi pada gambar.
   * **PCA (Principal Component Analysis)**: Mereduksi dimensi piksel menjadi komponen utama penting untuk mempercepat latihan.
   * **Flatten Pixel**: Menggunakan representasi vektor 1D piksel mentah langsung.

4. **Metode Pembelajaran Mesin Interaktif (Langkah 4)**:
   * **SVM (Support Vector Machine)** dengan kernel linear.
   * **KNN (K-Nearest Neighbors)** dengan slider jumlah tetangga ($k$) dinamis.
   * **Random Forest** dengan slider jumlah pohon ($n\_estimators$) dinamis.

5. **Evaluasi Model Komprehensif**:
   * **Laporan Klasifikasi**: Tabel detail metrik *precision*, *recall*, dan *f1-score* untuk masing-masing kelas angka (0-9).
   * **Confusion Matrix**: Matriks kebingungan interaktif yang dirender menggunakan Seaborn bertema gelap.

6. **Browser Restart Protection (Session Persistence)**:
   * Mengamankan progres kerja Anda! Jika browser di-refresh, tab tertutup secara tidak sengaja, atau koneksi terputus, aplikasi tidak akan mengulang dari Langkah 1 melainkan melanjutkan dari langkah terakhir lengkap dengan data yang sudah di-load atau model yang sudah dilatih.
   * Menggunakan penyimpanan `pickle` aman yang dialokasikan di direktori temporary Drive C untuk mencegah kegagalan penyimpanan akibat disk penuh di Drive D.

7. **Desain Visual Premium**:
   * Tampilan modern dengan Google Fonts (Outfit), efek kartu kaca (glassmorphism), hover tombol melayang dengan pendaran cahaya (glow), dan tombol navigasi simetris yang rapi.

---

## 📂 Struktur Direktori Proyek

```text
├── app/
│   ├── components/
│   │   ├── __init__.py
│   │   └── feature_extraction.py  # Modul ekstraksi fitur (HOG, PCA)
│   ├── DS-2/                      # Direktori dataset (Diunduh terpisah)
│   └── app.py                     # File utama aplikasi Streamlit
├── .gitignore                     # Konfigurasi file yang diabaikan oleh Git
├── requirements.txt               # Daftar dependensi package Python
└── README.md                      # Dokumentasi proyek (file ini)
```

---

## 🛠️ Panduan Instalasi & Penggunaan Lokal

### Prasyarat
Pastikan komputer Anda sudah terinstal **Python 3.11 atau lebih baru**.

### Langkah-langkah
1. **Clone Repositori**:
   ```bash
   git clone https://github.com/USERNAME-ANDA/NAMA-REPOSITORI.git
   cd NAMA-REPOSITORI
   ```

2. **Buat Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Aktifkan Virtual Environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   * **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Instal Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Jalankan Aplikasi**:
   ```bash
   streamlit run app/app.py
   ```
   Aplikasi secara otomatis akan terbuka di browser Anda pada alamat `http://localhost:8501`.

---

## 📊 Penyiapan & Unduhan Dataset

Karena ukuran dataset gambar yang cukup besar, folder dataset **tidak dimasukkan** ke dalam repositori GitHub ini (diabaikan menggunakan `.gitignore`).

### 1. Unduh Dataset via Google Drive
Anda dapat mengunduh zip dataset yang digunakan dalam proyek ini melalui tautan berikut:
* 📥 **[Unduh Dataset di Google Drive (Ganti dengan Link Anda)](https://drive.google.com/...)**

### 2. Cara Penempatan Folder
1. Ekstrak berkas `.zip` dataset yang telah diunduh.
2. Tempatkan folder hasil ekstrak (misal: `DS-2`) di dalam direktori `app/` sehingga strukturnya menjadi `app/DS-2/`.
3. Jalankan aplikasi Streamlit dan masukkan path lokal tersebut di Langkah ke-1.

*(Catatan: Anda juga bisa melewati pengunduhan lokal ini dan menggunakan tab **"Seret & Lepas File"** di Langkah ke-1 untuk mengunggah gambar langsung dari komputer Anda).*

### 3. Format Penamaan Berkas Gambar
Jika ingin menambahkan gambar dataset kustom Anda sendiri, pastikan berkas gambar dinamai menggunakan format berikut agar label angka dapat dibaca secara otomatis saat dimuat:

$$\text{\{label\_angka\}}\_\text{\{id\_unik\}}.[\text{png}|\text{jpg}|\text{jpeg}]$$

* Contoh Penamaan:
  - `3_001.png` (Gambar angka 3, ID unik 001)
  - `3_002.jpg` (Gambar angka 3, ID unik 002)
  - `7_handwritten_05.png` (Gambar angka 7, ID unik handwritten_05)
  - `0_img_99.png` (Gambar angka 0, ID unik img_99)

Sistem akan secara otomatis memotong nama berkas berdasarkan karakter garis bawah (`_`) pertama untuk mengenali target label angka.

---

## 🛡️ Lisensi & Kontributor

Proyek ini dibuat untuk tujuan pembelajaran machine learning dan pengolahan citra digital. Silakan modifikasi, sebarkan, dan gunakan proyek ini sesuai kebutuhan Anda!
