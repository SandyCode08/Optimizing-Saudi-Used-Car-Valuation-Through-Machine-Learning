
# Optimizing Saudi Used Car Valuation through Machine Learning

Proyek ini memprediksi **harga mobil bekas di Saudi Arabia** menggunakan machine learning. Model yang digunakan XGBoost lengkap dengan feature engineering, hyperparameter tuning, dan interpretasi fitur menggunakan SHAP untuk estimasi harga yang akurat dan data-driven.

## Struktur Folder / File

```
Saudi_Used_Car_Capstone_3_Regression.ipynb   # Notebook eksplorasi, training, evaluasi, SHAP
app_saudi_used_car.py                        # Aplikasi Streamlit untuk prediksi harga mobil
custom_transformers.py                        # Script custom transformers
data_saudi_used_cars.csv                      # Dataset utama
Saudi_Used_Car_Price_Estimator.joblib        # Model hasil training
Saudi_Used_Car_Price_Estimator.pkl           # Versi pickle dari model
requirements.txt                              # List library yang dibutuhkan
__pycache__/                                  # Folder cache Python (bisa di-ignore)
```

## Cara Menjalankan

1. Pastikan **Python** dan library yang diperlukan sudah terinstall
Install dengan perintah:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn streamlit

2. Buka terminal / command prompt.

3. Pindah ke folder project, misal:
cd "C:\Users\hp\Downloads\Project"

4. Jalankan aplikasi Streamlit:  
```bash
streamlit run app_saudi_used_car.py
```

> Jika tidak menggunakan virtual environment, pastikan semua library yang dibutuhkan sudah terinstall.

---

##Fitur Aplikasi

- Prediksi harga mobil bekas di Saudi Arabia.

- Input user: Type, Region, Make, Gear Type, Origin, Options, Year, Engine Size, Mileage, Negotiable.

- Output: Perkiraan harga mobil secara cepat dan interaktif.
---

## Dataset

- File utama: `data_saudi_used_cars.csv`  
- Fitur:
  - `Type`       : Jenis mobil (SUV, Sedan, Hatchback, dll.)  
  - `Region`     : Lokasi/daerah penjualan mobil  
  - `Make`       : Merek mobil  
  - `Gear_Type`  : Jenis transmisi (Manual/Automatic)  
  - `Origin`     : Negara asal mobil  
  - `Options`    : Fitur tambahan/options mobil  
  - `Year`       : Tahun produksi mobil  
  - `Engine_Size`: Kapasitas mesin (liter)  
  - `Mileage`    : Jarak tempuh mobil (km)  
  - `Negotiable` : Status harga bisa dinegosiasi (True/False)  
  - `Price`      : Harga mobil (target)  

---

## Notebook

- `Saudi_Used_Car_Capstone_3_Regression.ipynb` berisi eksplorasi data, training model, evaluasi performa, dan analisis SHAP.

---

## Model

- Model tersimpan di file:  
  - `Saudi_Used_Car_Price_Estimator.joblib`  
  - `Saudi_Used_Car_Price_Estimator.pkl`  

---

## Kontak

Untuk pertanyaan atau masukan, silakan hubungi **Author**.
