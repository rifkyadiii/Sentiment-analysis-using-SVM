# Analisis Sentimen Menggunakan Support Vector Machine (SVM)

Nama            : Moch Rifky Aulia Adikusumah
NIM             : 1227050072
Kelas           : Praktikum Pembelajaran Mesin E
Referensi Tugas : [https://github.com/jatinwarade/Sentiment-analysis-using-SVM]

## Deskripsi
Proyek ini mengimplementasikan model analisis sentimen menggunakan algoritma Support Vector Machine (SVM) untuk mengklasifikasikan sentimen dalam teks. Model ini dilatih untuk memprediksi apakah suatu teks mengandung sentimen positif atau negatif berdasarkan dataset ulasan yang telah dilabeli.

## Kebutuhan Sistem
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- nltk
- textblob
- seaborn
- matplotlib

## Instalasi
```bash
pip install pandas numpy scikit-learn nltk textblob seaborn matplotlib jupyter
```

Setelah instalasi, pastikan untuk mengunduh data tambahan yang diperlukan untuk NLTK:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Dataset
Dataset yang digunakan dalam proyek ini di ambil dari Kaggle UMICH S1650 [https://www.kaggle.com/c/si650winter11/data]

Program menggunakan dataset dalam format teks tab-separated (`training.txt`) dengan dua kolom:
1. `liked` - Label sentimen (misalnya: 0 untuk negatif, 1 untuk positif)
2. `text` - Teks ulasan

## Fitur Utama
1. **Eksplorasi Data (EDA)**
   - Menghitung jumlah data
   - Statistik deskriptif berdasarkan kategori sentimen

2. **Pra-pemrosesan Teks**
   - Tokenisasi menggunakan TextBlob
   - Lemmatisasi (mengubah kata ke bentuk dasar)

3. **Ekstraksi Fitur**
   - Bag of Words menggunakan CountVectorizer
   - Transformasi TF-IDF untuk memberikan bobot pada kata-kata penting

4. **Pemodelan dengan SVM**
   - Pembuatan pipeline untuk preprocessing dan klasifikasi
   - Optimasi hyperparameter menggunakan GridSearchCV
   - Validasi silang dengan StratifiedKFold

5. **Evaluasi Model**
   - Classification report (precision, recall, f1-score)
   - Contoh prediksi untuk teks baru

6. **Implementasi Kernel Gaussian**
   - Fungsi untuk menghitung nilai kernel Gaussian antara dua vektor

## Cara Penggunaan
1. Siapkan dataset `training.txt` dalam format yang benar
2. Jalankan notebook Jupyter (`jupyter notebook`)
3. Buka file notebook dan jalankan sel secara berurutan

## Contoh Prediksi
```python
# Prediksi untuk kalimat positif
classifier.predict(["the vinci code is awesome"])

# Prediksi untuk kalimat negatif
classifier.predict(["the vinci code is bad"])
```

## Struktur Proyek
```
├── SVM.ipynb                    # Notebook utama
├── training.txt                 # Dataset
├── requirements.txt             # Daftar Library yang digunakan
└── README.md                    # Dokumentasi proyek
```

## Metode dan Pendekatan
Proyek ini menggunakan pendekatan machine learning klasik untuk analisis sentimen:
1. **Ekstraksi Fitur**: Mengubah teks menjadi vektor fitur menggunakan representasi Bag of Words dan TF-IDF
2. **Klasifikasi**: Menggunakan Support Vector Machine (SVM) dengan optimasi parameter melalui grid search
3. **Validasi**: Menggunakan validasi silang 5-fold untuk memastikan generalisasi model

## Kontribusi
Kontribusi dipersilakan melalui pull request. Untuk perubahan besar, harap buka issue terlebih dahulu untuk mendiskusikan perubahan yang diinginkan.

## Lisensi
[MIT License](https://opensource.org/licenses/MIT)
