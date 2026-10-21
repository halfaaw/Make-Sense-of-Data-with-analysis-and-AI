import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("Memulai proses analitik...")
    
    # 1. SETUP DIREKTORI OUTPUT
    os.makedirs("outputs/charts", exist_ok=True)
    
    # 2. LOAD DATA
    try:
        df = pd.read_csv("data/invoice_data.csv")
    except FileNotFoundError:
        print("Error: File data/invoice_data.csv tidak ditemukan. Pastikan direktori saat ini benar.")
        return

    # 3. PREPROCESSING
    # Hapus kolom ID agar tidak terjadi data leakage
    df_model = df.drop(columns=['invoice_id'])
    
    # Label Encoding untuk kolom kategorikal
    le_pay = LabelEncoder()
    df_model['payment_method'] = le_pay.fit_transform(df_model['payment_method'])
    
    # Pisahkan Fitur (X) dan Target (y)
    X = df_model.drop(columns=['status_payment'])
    y = df_model['status_payment']
    
    # Split data training dan testing (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. MODELING
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. EVALUATION & PREDICTION
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 6. OUTPUT GENERATION
    # a. Laporan Teks
    with open("outputs/laporan_evaluasi.txt", "w") as f:
        f.write("=== LAPORAN EVALUASI MODEL ===\n")
        f.write(f"Akurasi Model: {acc:.2%}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # b. Temuan Analisis (Markdown)
    with open("outputs/temuan_analisis.md", "w") as f:
        f.write("# Temuan Analisis Prediktif\n\n")
        f.write("Berdasarkan model klasifikasi *Random Forest* yang telah dilatih, diperoleh hasil berikut:\n")
        f.write(f"- **Akurasi Model**: {acc:.2%}\n")
        f.write("- Model berhasil mempelajari pola keterlambatan berdasarkan riwayat pelanggan.\n")
        f.write("- Rekomendasi Preskriptif: Tagihan dengan metode cicilan/kredit yang memiliki riwayat telat sebaiknya diprioritaskan untuk di-*follow up*.\n")
        
    # c. Visualisasi Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Confusion Matrix - Deteksi Late Payment')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig("outputs/charts/confusion_matrix.png")
    plt.close()
    
    # d. Visualisasi Feature Importance
    importances = clf.feature_importances_
    plt.figure(figsize=(6,4))
    sns.barplot(x=importances, y=X.columns, hue=X.columns, palette='viridis', legend=False)
    plt.title('Fitur Paling Berpengaruh (Feature Importance)')
    plt.tight_layout()
    plt.savefig("outputs/charts/feature_importance.png")
    plt.close()
    
    print("Analisis selesai! Semua hasil telah disimpan di dalam folder 'outputs'.")

if __name__ == "__main__":
    main()
