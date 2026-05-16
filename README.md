# IF3270 Tubes 2 — Hidup Spinwheel

> **Tugas Besar 2 IF3270 Pembelajaran Mesin**
> Convolutional Neural Network dan Recurrent Neural Network

---

## Deskripsi Singkat

Repository ini berisi implementasi dua bagian besar dalam Tugas Besar 2 mata kuliah **IF3270 Pembelajaran Mesin**:

### Bagian 1 — Convolutional Neural Network (CNN)
Implementasi dan eksperimen CNN untuk **klasifikasi gambar** (6 kelas) menggunakan dataset Intel Image Classification. Eksperimen dilakukan dengan 16 kombinasi arsitektur yang bervariasi pada:
- Jumlah layer konvolusi: `2` atau `4`
- Banyak filter per layer: `[32, 64]` atau `[64, 128]`
- Ukuran filter (kernel): `3×3` atau `5×5`
- Jenis pooling: `MaxPooling` atau `AveragePooling`

Selain menggunakan Keras, setiap layer CNN juga diimplementasikan **from scratch** dengan NumPy, mencakup:
- `Conv2D` (parameter sharing) dan `LocallyConnected2D` (tanpa parameter sharing)
- `MaxPooling2D`, `AveragePooling2D`, `GlobalMaxPooling2D`, `GlobalAveragePooling2D`
- `Flatten`, `Dense` dengan berbagai fungsi aktivasi

### Bagian 2 — Image Captioning (RNN/LSTM)
Implementasi **image captioning** pada dataset Flickr8k menggunakan arsitektur encoder–decoder:
- **Encoder**: InceptionV3 (pretrained, frozen) untuk mengekstrak fitur gambar (2048-dim)
- **Decoder**: RNN atau LSTM dengan 1–3 layer dan 128/512 hidden units

Model dievaluasi menggunakan metrik **BLEU-4**.

---

## Struktur Repository

```
IF3270_Tubes2_HidupSpinwheel/
├── cnn/
│   ├── layers.py                     # Implementasi layer CNN from scratch (NumPy)
│   ├── cnn_scratch.py                # Kelas CNNScratch & CNNScratchLC
│   ├── train_keras.py                # Training 16 arsitektur CNN dengan Keras
│   ├── evaluate.py                   # Evaluasi, plotting, dan perbandingan model
│   ├── feature_extraction_flickr8k.py# Ekstraksi fitur InceptionV3 untuk Flickr8k
│   └── utils.py                      # Fungsi utilitas umum
├── rnn_lstm/
│   ├── rnn_layers.py                 # Implementasi layer RNN/LSTM from scratch
│   ├── captioning_scratch.py         # Model captioning from scratch
│   ├── train_captioning.py           # Training decoder RNN/LSTM dengan Keras
│   └── evaluate_captioning.py        # Evaluasi BLEU-4 dan visualisasi caption
├── model/
│   └── preprocess.py                 # Preprocessing data
├── notebook/
│   ├── main.ipynb                    # Notebook utama (eksperimen & analisis CNN)
│   └── Analisis.ipynb                # Notebook analisis tambahan (FINAL)
├── doc/                              # Folder dokumentasi
├── requirements.txt                  # Dependensi Python
└── README.md
```

---

## Setup dan Instalasi

### Prasyarat
- Python **3.9 – 3.11** (disarankan)
- pip
- GPU opsional (sangat disarankan untuk training)

### 1. Clone Repository

```bash
git clone https://github.com/DitaMaheswari05/IF3270_Tubes2_HidupSpinwheel.git
cd IF3270_Tubes2_HidupSpinwheel
```

### 2. Buat Virtual Environment (disarankan)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset

#### Intel Image Classification (untuk CNN)
Download dataset dari [Kaggle – Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dan ekstrak ke folder yang sesuai.

#### Flickr8k (untuk Image Captioning)
Download dataset dari [Kaggle – Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dan ekstrak. Struktur yang dibutuhkan:
```
<flickr8k_root>/
├── Images/        # semua gambar .jpg
└── captions.txt   # file caption
```

### 5. Hasil Model

Karena kami menjalankan model dan analisis di Google Colab, kami menyimpan hasil file bobot model terbaik (`.keras`) di Google Drive:
```
https://drive.google.com/file/d/1M9iAYI4n3jK278SUpSXxQDnz1rfr_GXG/view?usp=sharing
```

---

## Cara Menjalankan

```bash
jupyter notebook
```
Buka `notebook/analisis.ipynb` dan jalankan sel secara berurutan.

---

## Hasil Eksperimen

Hasil evaluasi lengkap (confusion matrix, learning curves, perbandingan model Keras vs. from scratch, skor BLEU-4) dapat dilihat di:
- `notebook/main.ipynb` — eksperimen utama
- `notebook/Analisis.ipynb` — analisis perbandingan

---

## Pembagian Tugas

| No | Nama | NIM | Tugas |
|----|------|-----|-------|
| 1  | Dita Maheswari | 13523125 | Notebook, Laporan, Implementasi RNN & LSTM |
| 2  | Boye Mangaratua Ginting | 13523127| Notebook, Laporan, Implementasi RNN & LSTM |
| 3  | Samantha Laqueenna Ginting| 13523138 | Notebook, Laporan, Implementasi CNN |

---