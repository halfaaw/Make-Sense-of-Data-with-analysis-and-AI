# Make-Sense-of-Data-with-analysis-and-AI
# Studi Kasus Sesi 1: Prediksi Keterlambatan Pembayaran (Late Payment) PT Maju Bersama

## 1. Latar Belakang
PT Maju Bersama (perusahaan fiktif) mencatat kenaikan omzet, namun di saat yang sama keuntungan bersih menurun dan piutang (receivables) membengkak karena banyaknya pelanggan yang telat membayar tagihan. Studi kasus ini membuat alur analitik prediktif yang dapat diulang menggunakan Python untuk mendeteksi potensi tagihan macet.

Keterkaitan dengan seminar: Proyek ini menerapkan *Predictive Analytics* (klasifikasi) menggunakan metodologi CRISP-DM yang dibahas oleh narasumber. Proyek ini juga menekankan pentingnya *data preparation* dan penghapusan fitur yang bisa menyebabkan kebocoran data (*data leakage*), karena data yang buruk akan menghasilkan model prediksi yang keliru (*garbage in, garbage out*).

## 2. Tujuan
- Memvalidasi dan menyiapkan data tagihan (*invoice*) sebelum pemodelan.
- Melakukan transformasi data kategorikal menjadi numerik (*encoding*).
- Melatih model *Machine Learning* untuk memprediksi probabilitas keterlambatan pembayaran.
- Menghasilkan evaluasi model seperti akurasi dan *confusion matrix* yang bisa diperiksa kembali.
- Menuliskan batasan analisis bahwa AI hanya bertindak sebagai asisten, sementara keputusan penagihan tetap berada di tangan manusia.

## 3. Data
File `data/invoice_data.csv` berisi data sintetis yang dibuat khusus untuk latihan, bukan data transaksi perusahaan sungguhan. Setiap baris adalah satu catatan *invoice*.

| Kolom | Arti |
|---|---|
| `invoice_id` | ID unik untuk setiap tagihan |
| `invoice_amount` | Total nilai tagihan dalam rupiah |
| `payment_method` | Metode pembayaran (misal: Transfer, Credit, Cash) |
| `customer_history_late` | Jumlah riwayat pelanggan telat bayar di masa lalu |
| `status_payment` | Status aktual pembayaran (Late / Not Late) sebagai target |

**Konsep yang digunakan:**
- ID Tagihan (`invoice_id`) dihapus sebelum pemodelan untuk menghindari *overfitting* atau *data leakage*.
- Evaluasi akurasi = (Prediksi Benar) ÷ (Total Data) × 100%
- *Confusion Matrix* digunakan untuk membedah seberapa banyak tagihan telat yang gagal diprediksi oleh model.

## 4. Struktur Proyek
```text
studi-kasus-sesi-1-analisis-penjualan/
├── data/
│   └── invoice_data.csv
├── src/
│   └── prediksi_pembayaran.py
├── tests/
│   └── test_prediksi_pembayaran.py
├── outputs/
│   └── charts/             # grafik evaluasi dibuat ketika program dijalankan
├── README.md
└── requirements.txt
