# 📊 Prediksi Keterlambatan Pembayaran Piutang Dagang (Late Payment)
### Studi Kasus: PT Maju Bersama

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Framework](https://img.shields.io/badge/Framework-CRISP--DM-green)

Proyek portofolio analitik data yang memprediksi **faktur penjualan kredit mana yang berisiko dibayar terlambat** menggunakan algoritma **Random Forest Classifier**, dengan kerangka kerja **CRISP-DM**. Pendekatan ini menjembatani sudut pandang **akuntansi (manajemen piutang)** dan **data science**.

> ⚠️ **Catatan:** Dataset adalah **data dummy** (20 baris) untuk tujuan demonstrasi. Hasil metrik tidak merepresentasikan performa di dunia nyata.

---

## 📑 Daftar Isi
1. [Business Understanding](#1--business-understanding)
2. [Data Understanding](#2--data-understanding)
3. [Data Preparation](#3--data-preparation)
4. [Modeling](#4--modeling)
5. [Insight & Tindakan Preskriptif](#5--insight--tindakan-preskriptif)
6. [Cara Menjalankan Kode](#6--cara-menjalankan-kode)
7. [Struktur Repositori](#7--struktur-repositori)

---

## 1. 🎯 Business Understanding

### Masalah Bisnis
PT Maju Bersama menghadapi dua gejala yang saling berkaitan:
- 📉 **Laba menurun**, karena kas tertahan dan biaya penagihan meningkat.
- 📈 **Saldo piutang dagang meningkat**, karena banyak faktur dibayar melewati jatuh tempo.

Keterlambatan pembayaran memperlambat arus kas masuk, menaikkan risiko piutang tak tertagih, dan menekan likuiditas operasional. Selama ini penagihan bersifat **reaktif** (menagih setelah jatuh tempo).

### Tujuan Analitik
Membangun model klasifikasi yang **memprediksi faktur berisiko telat sebelum jatuh tempo**, sehingga tim keuangan dapat bertindak **proaktif** dan memprioritaskan penagihan.

| Aspek | Uraian |
|---|---|
| **Target** | `Status_Late` (1 = telat, 0 = tepat waktu) |
| **Jenis masalah** | Klasifikasi biner |
| **Pengguna hasil** | Tim Akuntansi Piutang, Kredit & Penagihan, Manajer Keuangan |

---

## 2. 🔍 Data Understanding

| Kolom | Deskripsi |
|---|---|
| `ID_Faktur` | Nomor unik faktur |
| `Kategori_Klien` | Corporate / Retail / Pemerintah |
| `Nilai_Faktur` | Nilai tagihan (Rupiah) |
| `Termin_Pembayaran` | Net 14 / Net 30 / Net 60 |
| `Rasio_Keterlambatan_Historis` | Persentase keterlambatan klien di masa lalu |
| `tanggal_bayar_aktual` | ⚠️ Tanggal pembayaran sebenarnya (*leakage*) |
| `jumlah_hari_telat` | ⚠️ Jumlah hari keterlambatan (*leakage*) |
| `Status_Late` | 🎯 Variabel target |

---

## 3. 🧹 Data Preparation

### Mengapa `tanggal_bayar_aktual` dan `jumlah_hari_telat` dihapus?

Kedua kolom ini adalah sumber **data leakage**:

- Informasi tersebut **baru tersedia setelah klien membayar**, sedangkan model harus memprediksi **saat faktur terbit**.
- `Status_Late` secara langsung diturunkan dari `jumlah_hari_telat` (> 0 berarti telat). Jika kolom ini dipakai, model hanya "membaca jawaban".
- Akibatnya akurasi tampak mendekati sempurna, tetapi **model gagal total saat dipakai pada faktur baru**.

Langkah persiapan lainnya:
- Menghapus `ID_Faktur` (hanya pengenal, tanpa makna prediktif).
- **Ordinal encoding** untuk `Termin_Pembayaran` (14 < 30 < 60 hari).
- **One-hot encoding** untuk `Kategori_Klien` (kategori tanpa urutan).
- Pembagian data **80% latih / 20% uji** dengan *stratified split*.

---

## 4. 🤖 Modeling

### Mengapa Random Forest Classifier?

| Alasan | Penjelasan |
|---|---|
| ✅ Cocok untuk klasifikasi biner | Memprediksi telat vs tepat waktu |
| ✅ Tahan terhadap *overfitting* | Gabungan banyak pohon keputusan (*ensemble*) |
| ✅ Menangkap pola non-linear | Interaksi antara nilai faktur, termin, dan riwayat klien |
| ✅ Tidak perlu penskalaan fitur | Pohon keputusan tidak sensitif terhadap skala angka |
| ✅ Interpretabel | Menyediakan *Feature Importance* untuk dijelaskan ke manajemen |

**Metrik evaluasi:** Accuracy, Precision, dan Recall.
Dalam konteks piutang, **Recall** penting karena kita ingin menangkap sebanyak mungkin faktur yang benar-benar berisiko telat.

---

## 5. 💡 Insight & Tindakan Preskriptif

Rekomendasi tata kelola keuangan berbasis hasil model (*Feature Importance* dan skor risiko per faktur):

| Temuan | Tindakan Preskriptif |
|---|---|
| Rasio keterlambatan historis tinggi menjadi pemicu utama | Terapkan **limit kredit dinamis** dan tinjau ulang klien berisiko tinggi secara berkala |
| Termin panjang (Net 60) cenderung berisiko lebih tinggi | **Perketat termin** (Net 60 menjadi Net 30) untuk klien berisiko, atau beri **diskon pembayaran dini** |
| Faktur bernilai besar yang diprediksi telat | **Pendampingan klien**: konfirmasi proaktif, jadwal pembayaran, eskalasi ke manajer akun |
| Klien kategori tertentu sering telat | Siapkan **SOP penagihan khusus** per kategori (misalnya koordinasi administrasi untuk klien pemerintah) |
| Faktur berisiko tinggi terdeteksi sejak terbit | Jalankan **pengingat bertahap**: H-7, H-3, H-0 jatuh tempo |

**Dampak yang diharapkan:** percepatan arus kas, penurunan saldo piutang macet, dan perbaikan laba melalui berkurangnya beban penagihan dan cadangan kerugian piutang.

---

## 6. ▶️ Cara Menjalankan Kode

```bash
# 1. Clone repositori
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan skrip
python prediksi_late_payment.py
```

---

## 7. 📁 Struktur Repositori

```
├── data_invoice_maju_bersama.csv   # Dataset dummy
├── prediksi_late_payment.py        # Skrip analisis & pemodelan
├── requirements.txt                # Dependensi Python
└── README.md                       # Dokumentasi proyek
```

---

## 🚀 Pengembangan Lanjutan
- Tambah data nyata dalam jumlah lebih besar (ratusan hingga ribuan faktur).
- Gunakan **cross-validation** dan **hyperparameter tuning**.
- Tambah fitur: umur piutang, musim penagihan, frekuensi transaksi klien.
- Terapkan penanganan ketidakseimbangan kelas (misalnya SMOTE) bila perlu.

---

👤 **Author:** <Nama Anda> | 🔗 [LinkedIn](#) | 📧 email@contoh.com
