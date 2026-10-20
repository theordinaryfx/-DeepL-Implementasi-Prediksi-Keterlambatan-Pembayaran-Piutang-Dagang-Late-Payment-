"""
Prediksi Keterlambatan Pembayaran Piutang Dagang (Late Payment)
PT Maju Bersama - Pendekatan CRISP-DM dengan Random Forest Classifier

Alur: Data Understanding -> Data Preparation -> Modeling -> Evaluation
"""

# ---------------------------------------------------------------
# 1. IMPORT LIBRARY
# ---------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ---------------------------------------------------------------
# 2. MEMUAT DATASET (Data Understanding)
# ---------------------------------------------------------------
df = pd.read_csv("data_invoice_maju_bersama.csv")

print("=== 5 Baris Pertama Data ===")
print(df.head())
print("\n=== Info Dataset ===")
print(df.info())
print("\n=== Distribusi Target (Status_Late) ===")
print(df["Status_Late"].value_counts())

# ---------------------------------------------------------------
# 3. PEMBERSIHAN DATA (Data Preparation)
# ---------------------------------------------------------------
# PENTING - MENCEGAH DATA LEAKAGE:
# 'tanggal_bayar_aktual' dan 'jumlah_hari_telat' baru diketahui SETELAH
# pembayaran terjadi. Saat faktur diterbitkan, informasi ini belum ada,
# padahal model harus memprediksi SEBELUM pembayaran. Jika dipakai,
# model akan "mengintip jawaban" dan akurasinya palsu.
kolom_leakage = ["tanggal_bayar_aktual", "jumlah_hari_telat"]
df = df.drop(columns=kolom_leakage)

# 'ID_Faktur' hanyalah pengenal unik, tidak punya makna prediktif.
df = df.drop(columns=["ID_Faktur"])

# Cek nilai kosong (jika ada, baris dibuang agar sederhana)
print("\nJumlah nilai kosong per kolom:\n", df.isnull().sum())
df = df.dropna()

# Pengaman: pastikan kolom leakage benar-benar sudah tidak ada
assert not set(kolom_leakage) & set(df.columns), "Kolom leakage masih ada!"

# ---------------------------------------------------------------
# 4. ENCODING VARIABEL KATEGORIKAL
# ---------------------------------------------------------------
# Termin_Pembayaran bersifat ORDINAL (urutan: 14 < 30 < 60 hari),
# sehingga dipetakan ke angka hari sesuai maknanya.
df["Termin_Pembayaran"] = (
    df["Termin_Pembayaran"].str.replace("Net ", "", regex=False).astype(int)
)

# Kategori_Klien bersifat NOMINAL (tidak ada urutan), maka memakai
# One-Hot Encoding. drop_first=True menghindari kolom yang redundan.
df = pd.get_dummies(df, columns=["Kategori_Klien"], drop_first=True, dtype=int)

# ---------------------------------------------------------------
# 5. PEMISAHAN FITUR (X) DAN TARGET (y), LALU TRAIN-TEST SPLIT
# ---------------------------------------------------------------
X = df.drop(columns=["Status_Late"])  # fitur prediktor
y = df["Status_Late"]                  # target: 1 = telat, 0 = tepat waktu

# 80% data latih, 20% data uji.
# stratify=y menjaga proporsi telat/tepat waktu tetap seimbang di kedua set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData latih: {X_train.shape[0]} baris | Data uji: {X_test.shape[0]} baris")

# ---------------------------------------------------------------
# 6. PELATIHAN MODEL (Modeling)
# ---------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,  # jumlah pohon keputusan
    random_state=42,   # agar hasil dapat direproduksi
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 7. EVALUASI MODEL (Evaluation)
# ---------------------------------------------------------------
y_pred = model.predict(X_test)

# zero_division=0 mencegah peringatan jika model tidak memprediksi kelas 1
print("\n=== Hasil Evaluasi Model ===")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.2f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.2f}")

# ---------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------------
# Menunjukkan fitur mana yang paling berpengaruh terhadap prediksi.
importance = (
    pd.Series(model.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)
print("\n=== Feature Importance ===")
print(importance.round(4))

# Catatan: dataset hanya 20 baris (data dummy), sehingga data uji hanya
# 4 baris. Metrik di atas bersifat ilustratif, bukan bukti performa nyata.
