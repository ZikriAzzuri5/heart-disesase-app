# ❤️ Aplikasi Prediksi Risiko Penyakit Jantung

Aplikasi web interaktif berbasis **Streamlit** yang memanfaatkan model **Machine Learning (Logistic Regression)** untuk memprediksi tingkat risiko penyakit jantung berdasarkan data medis dan gaya hidup pasien.

> ⚠️ **Disclaimer**: Aplikasi ini hanya berfungsi sebagai alat bantu skrining awal dan **tidak menggantikan diagnosis medis resmi** oleh dokter.

---

## 📋 Daftar Isi

- [Fitur Aplikasi](#-fitur-aplikasi)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi & Menjalankan Aplikasi](#-instalasi--menjalankan-aplikasi)
- [Parameter Input](#-parameter-input)
- [Cara Kerja Aplikasi](#-cara-kerja-aplikasi)
- [Contoh Penggunaan](#-contoh-penggunaan)

---

## ✨ Fitur Aplikasi

- 🩺 **Form input interaktif** dengan 28 parameter kesehatan (demografi, gaya hidup, dan data klinis)
- 📊 **Prediksi real-time** dengan estimasi probabilitas risiko dalam persen (%)
- 🎨 **Tampilan visual yang informatif** dengan indikator warna (merah = risiko tinggi, hijau = risiko rendah)
- ⚡ **Performa cepat** berkat caching model menggunakan `@st.cache_resource`
- 📱 **Responsive layout** dengan tampilan dua kolom yang rapi

---

## 🛠 Teknologi yang Digunakan

| Teknologi | Versi | Kegunaan |
| :--- | :--- | :--- |
| Python | 3.11+ | Bahasa pemrograman utama |
| Streamlit | ≥ 1.20.0 | Framework web UI interaktif |
| Scikit-learn | ≥ 1.0.0 | Machine Learning (Logistic Regression & StandardScaler) |
| Pandas | ≥ 1.5.0 | Manipulasi data tabular (DataFrame) |
| NumPy | ≥ 1.21.0 | Komputasi numerik |
| Joblib | (built-in) | Memuat file model & scaler (.pkl) |

---

## 📁 Struktur Proyek

```
heart-disease-app/
│
├── app.py                      # Kode utama aplikasi Streamlit
├── heart_disease_model.pkl     # Model Logistic Regression (pre-trained)
├── heart_disease_scaler.pkl    # StandardScaler (pre-fitted)
├── requirements.txt            # Daftar dependensi Python
└── README.md                   # Dokumentasi proyek (file ini)
```

### Penjelasan File Utama

| File | Deskripsi |
| :--- | :--- |
| `app.py` | Kode utama yang berisi UI Streamlit, preprocessing data, dan logika prediksi |
| `heart_disease_model.pkl` | Model Logistic Regression yang sudah dilatih untuk klasifikasi risiko penyakit jantung (output: 0 = Risiko Rendah, 1 = Risiko Tinggi) |
| `heart_disease_scaler.pkl` | StandardScaler yang sudah di-fit untuk menormalisasi 28 fitur input agar sesuai dengan skala yang diharapkan model |
| `requirements.txt` | Daftar library Python yang diperlukan untuk menjalankan aplikasi |

---

## 🚀 Instalasi & Menjalankan Aplikasi

### 1. Clone / Download Proyek

```bash
git clone <repository-url>
cd heart-disease-app
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat `http://localhost:8501`.

---

## 📊 Parameter Input

Aplikasi menerima **28 parameter** yang dibagi menjadi 2 kategori:

### 👤 Informasi Demografi & Gaya Hidup

| No | Parameter | Tipe | Rentang |
| :---: | :--- | :--- | :--- |
| 1 | Umur (Tahun) | Angka | 18 - 120 |
| 2 | Jenis Kelamin | Pilihan | Laki-laki / Perempuan |
| 3 | Indeks Massa Tubuh (BMI) | Desimal | 10.0 - 60.0 |
| 4 | Status Merokok | Pilihan | Tidak Pernah / Mantan / Aktif |
| 5 | Konsumsi Alkohol (Unit/Minggu) | Desimal | 0.0 - 100.0 |
| 6 | Olahraga (Menit/Minggu) | Angka | 0 - 1000 |
| 7 | Langkah Harian | Angka | 0 - 50.000 |
| 8 | Jam Tidur per Hari | Desimal | 1.0 - 24.0 |
| 9 | Skor Stres | Slider | 0 - 100 |
| 10 | Skor Kualitas Diet | Slider | 0 - 100 |
| 11 | Menggunakan Smartwatch | Ya/Tidak | - |
| 12 | Riwayat Keluarga Penyakit Jantung | Ya/Tidak | - |

### 🩺 Pemeriksaan Klinis & Laboratorium

| No | Parameter | Tipe | Rentang |
| :---: | :--- | :--- | :--- |
| 13 | Tekanan Darah Sistolik (mmHg) | Angka | 70 - 250 |
| 14 | Tekanan Darah Diastolik (mmHg) | Angka | 40 - 150 |
| 15 | Kolesterol Total (mg/dL) | Angka | 80 - 600 |
| 16 | Kolesterol HDL (mg/dL) | Angka | 10 - 150 |
| 17 | Kolesterol LDL (mg/dL) | Angka | 20 - 400 |
| 18 | Trigliserida (mg/dL) | Angka | 30 - 1000 |
| 19 | Gula Darah Puasa (mg/dL) | Angka | 50 - 500 |
| 20 | HbA1c (%) | Desimal | 3.0 - 18.0 |
| 21 | Detak Jantung Istirahat (bpm) | Angka | 40 - 200 |
| 22 | Detak Jantung Maksimum (bpm) | Angka | 60 - 230 |
| 23 | Tipe Nyeri Dada | Pilihan | Asymptomatic / Atypical / Non-Anginal / Typical |
| 24 | Nyeri Dada Saat Olahraga | Ya/Tidak | - |
| 25 | ST Depression (mm) | Desimal | 0.0 - 10.0 |

---

## ⚙️ Cara Kerja Aplikasi

Aplikasi ini bekerja dalam 4 tahap utama:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. INPUT    │───▶│ 2. ENCODING  │───▶│  3. SCALING  │───▶│ 4. PREDIKSI  │
│  Form Data   │    │  One-Hot &   │    │  Standard    │    │  Logistic    │
│  Pasien      │    │  Binary      │    │  Scaler      │    │  Regression  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Tahap 1: Input Data
Pengguna mengisi form data medis dan gaya hidup melalui antarmuka web Streamlit.

### Tahap 2: Encoding (Preprocessing)
Data kategorikal diubah menjadi format numerik:
- **One-Hot Encoding**: Tipe nyeri dada (3 kolom biner) dan status merokok (2 kolom biner)
- **Binary Encoding**: Jenis kelamin, riwayat keluarga, nyeri dada saat olahraga, kepemilikan smartwatch

### Tahap 3: Scaling (Normalisasi)
`StandardScaler` menormalisasi seluruh 28 fitur menggunakan rumus z-score:
```
z = (x - rata-rata) / standar_deviasi
```
Hal ini memastikan semua fitur memiliki bobot yang adil meskipun memiliki satuan dan rentang angka yang berbeda.

### Tahap 4: Prediksi
`LogisticRegression` menganalisis data ter-skala dan menghasilkan:
- **Klasifikasi**: 0 (Risiko Rendah) atau 1 (Risiko Tinggi)
- **Probabilitas**: Persentase estimasi risiko penyakit jantung

---

## 💡 Contoh Penggunaan

1. Buka aplikasi di browser (`http://localhost:8501`)
2. Isi data pasien pada form di kolom kiri (demografi & gaya hidup) dan kolom kanan (data klinis)
3. Klik tombol **"🔍 Analisis Risiko Sekarang"**
4. Lihat hasil prediksi:
   - 🟢 **Kotak Hijau** — Risiko penyakit jantung relatif rendah
   - 🔴 **Kotak Merah** — Terindikasi risiko tinggi penyakit jantung
5. Progress bar menunjukkan estimasi probabilitas risiko dalam persen

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan pembelajaran Data Science.

---

<p align="center">
  Dibuat dengan ❤️ menggunakan <b>Python</b> & <b>Streamlit</b>
</p>
