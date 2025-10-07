# app_saudi_used_car.py
import streamlit as st
import pandas as pd
import joblib
from custom_transformers import CustomTransformer  
from custom_transformers import reduce_make, reduce_type, to_int_transform
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ======================
# Load pipeline
# ======================
try:
    model = joblib.load('Saudi_Used_Car_Price_Estimator.joblib')
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

# ======================
# Pilihan valid dari data training
# ======================
TYPE_OPTIONS = ['Corolla', 'Yukon', 'Range Rover', 'Optima', 'FJ', 'CX3',
       'Cayenne S', 'Sonata', 'Avalon', 'LS', 'C300', 'Land Cruiser',
       'Hilux', 'Tucson', 'Caprice', 'Sunny', 'Pajero', 'Azera', 'Focus',
       '5', 'Spark', 'Camry', 'Pathfinder', 'Accent', 'ML', 'Tahoe',
       'Yaris', 'Suburban', 'A', 'Altima', 'Traverse', 'Expedition',
       'Senta fe', 'Liberty', '3', 'X', 'Elantra', 'Land Cruiser Pickup',
       'VTC', 'Malibu', 'The 5', 'A8', 'Patrol', 'Grand Cherokee', 'SL',
       'Previa', 'SEL', 'Aveo', 'MKZ', 'Victoria', 'Datsun', 'Flex',
       'GLC', 'ES', 'Edge', '6', 'Escalade', 'Innova', 'Navara', 'H1',
       'G80', 'Carnival', 'Symbol', 'Camaro', 'Accord', 'Avanza',
       'Land Cruiser 70', 'Taurus', 'C5700', 'Impala', 'Optra', 'S',
       'Other', 'Cerato', 'Furniture', 'Murano', 'Explorer', 'LX',
       'Pick up', 'Charger', 'H6', 'BT-50', 'Hiace', 'Ranger', 'Fusion',
       'Rav4', 'Ciocca', 'CX9', 'Kona', 'Sentra', 'Sierra', 'Durango',
       'CT-S', 'Sylvian Bus', 'Navigator', 'Opirus', 'Marquis', 'The 7',
       'FX', 'Creta', 'D-MAX', 'CS35', 'The 3', 'Dyna', 'GLE', 'Sedona',
       'Prestige', 'CLA', 'Lumina', 'Vanquish', 'Sorento', 'Safrane',
       'Cores', 'Cruze', 'Prado', 'Cadenza', "D'max", 'Silverado', 'Rio',
       'Maxima', 'X-Trail', 'RX', 'Cressida', 'C', 'Seven', 'Cherokee',
       'Grand Marquis', 'H2', 'QX', 'Blazer', 'Wingle', 'Panamera',
       'Rush', 'The M', 'Genesis', 'E', 'K5', 'CS95', 'Cayenne Turbo S',
       'Civic', 'Echo Sport', 'Challenger', 'CL', 'Wrangler', 'A6',
       'Dokker', 'CX5', 'Mohave', 'Ghost', 'Copper', 'Veloster', 'G',
       'Jetta', 'IS', 'Thunderbird', 'Fluence', 'V7', 'Vego', 'Aurion',
       'Q', 'F3', 'UX', 'Beetle', 'F150', 'Acadia', 'EC7', 'Lancer',
       'Capture', 'Van R', 'Mustang', 'CS35 Plus', 'DB9', 'APV',
       'Kaptiva', 'Viano', 'Safari', 'Cadillac', 'CLS', 'Duster',
       'Platinum', 'Carenz', 'Emgrand', 'Z', 'Coupe S', 'Odyssey',
       'Terrain', 'Juke', 'Sportage', 'C200', 'Attrage', 'GS', 'X-Terra',
       'Azkarra', 'XF', 'Picanto', 'Armada', 'CT5', 'KICKS', 'Gran Max',
       'Cayman', 'Levante', 'Montero', '300', 'POS24', 'A3', 'Touareg',
       'Passat', 'Delta', 'H3', 'RX5', 'GS3', 'Coupe', 'New Yorker',
       'Cayenne Turbo', 'Colorado', 'Trailblazer', 'Vitara', 'Nativa',
       'Van', 'LF X60', 'Koleos', 'Defender', 'Abeka', 'H100',
       'Flying Spur', 'Pilot', 'L200', 'A7', 'Quattroporte', 'Bora',
       'Compass', 'Bus Urvan', 'Macan', 'Corolla Cross', 'GL', 'City',
       'DTS', 'Ertiga', 'Envoy', 'CT6', 'Fleetwood', 'Tiggo', 'GX', 'Q5',
       'A4', 'XJ', 'Echo', 'HS', 'Avalanche', 'MKX', 'Seltos', 'SRX',
       'RX8', 'SLK', '301', 'EC8', '3008', 'Suvana', 'Prius', 'Cayenne',
       'Eado', 'The 6', 'Royal', 'NX', 'Soul', 'CS75', 'H9', 'F-Pace',
       'Coolray', 'Maybach', 'CS85', 'Jimny', 'GC7', '360', 'A5', 'S300',
       'Superb', 'Ram', 'The 4', 'Grand Vitara', '500', 'Logan', '5008',
       'Tiguan', 'Golf', 'S5', '911', 'Boxer', 'Camargue', 'M', 'Daily',
       'Nitro', 'CRV', 'Mini Van', 'Pegas', 'L300', 'Coaster',
       'Discovery', 'Montero2', 'Bentayga', 'Z370', 'Bus County',
       'Stinger', 'SRT', 'CT4', 'F Type', 'CC', 'Koranado', 'ASX',
       'Carens', 'Crown', 'ِACTIS V80', 'XT5', 'Tuscani', '4Runner',
       'ATS', 'HRV', 'X7', 'Outlander', 'X40', 'Q7', 'ZS', 'G70',
       'Megane', 'Nexon', 'Power', 'B50', 'Town Car', '2', 'i40', 'RC',
       'Doblo', 'Bronco', 'Dzire', 'Avante', 'Z350', 'CX7', 'Countryman',
       'GTB 599 Fiorano', 'Prestige Plus', 'Terios', 'MKS', 'Milan',
       'Centennial', 'Dakota', 'Savana', 'S8']

REGION_OPTIONS = ['Abha', 'Riyadh', 'Hafar Al-Batin', 'Aseer', 'Makkah', 'Dammam',
       'Yanbu', 'Al-Baha', 'Jeddah', 'Hail', 'Khobar', 'Al-Ahsa', 'Jazan',
       'Al-Medina', 'Al-Namas', 'Tabouk', 'Taef', 'Qassim', 'Arar',
       'Jubail', 'Sabya', 'Al-Jouf', 'Najran', 'Wadi Dawasir', 'Qurayyat',
       'Sakaka', 'Besha']

MAKE_OPTIONS = ['Toyota', 'GMC', 'Land Rover', 'Kia', 'Mazda', 'Porsche',
       'Hyundai', 'Lexus', 'Chrysler', 'Chevrolet', 'Nissan',
       'Mitsubishi', 'Ford', 'MG', 'Mercedes', 'Jeep', 'BMW', 'Audi',
       'Lincoln', 'Cadillac', 'Genesis', 'Renault', 'Honda', 'Suzuki',
       'Zhengzhou', 'Dodge', 'HAVAL', 'INFINITI', 'Isuzu', 'Changan',
       'Aston Martin', 'Mercury', 'Great Wall', 'Other', 'Rolls-Royce',
       'MINI', 'Volkswagen', 'BYD', 'Geely', 'Victory Auto', 'Classic',
       'Jaguar', 'Daihatsu', 'Maserati', 'Hummer', 'GAC', 'Lifan',
       'Bentley', 'Chery', 'Peugeot', 'Foton', 'Škoda', 'Fiat', 'Iveco',
       'SsangYong', 'FAW', 'Tata', 'Ferrari']

GEAR_OPTIONS = ["Manual", "Automatic"]

ORIGIN_OPTIONS = ['Saudi', 'Gulf Arabic', 'Other', 'Unknown']

OPTIONS_OPTIONS = ['Standard', 'Full', 'Semi Full'] 

YEAR_MIN, YEAR_MAX = 1963, 2022
ENGINE_MIN, ENGINE_MAX = 1.0, 9.0
MILEAGE_MIN, MILEAGE_MAX = 100, 20000000

# ======================
# Judul Aplikasi
# ======================
st.title("🚗 Prediksi Harga Mobil Bekas (Saudi)")
st.markdown("""
Aplikasi ini memprediksi **harga mobil bekas** berdasarkan spesifikasi lengkap mobil.
""")

# ======================
# Input user
# ======================
st.sidebar.header("Masukkan Spesifikasi Mobil")

type_ = st.sidebar.selectbox("Type", TYPE_OPTIONS)
region = st.sidebar.selectbox("Region", REGION_OPTIONS)
make = st.sidebar.selectbox("Make", MAKE_OPTIONS)
gear_type = st.sidebar.selectbox("Gear Type", GEAR_OPTIONS)
origin = st.sidebar.selectbox("Origin", ORIGIN_OPTIONS)
options = st.sidebar.multiselect("Options", OPTIONS_OPTIONS)
year = st.sidebar.slider("Year", YEAR_MIN, YEAR_MAX, 2015)
engine_size = st.sidebar.number_input("Engine Size (liter)", ENGINE_MIN, ENGINE_MAX, 2.0)
mileage = st.sidebar.number_input("Mileage (km)", MILEAGE_MIN, MILEAGE_MAX, 50000)
negotiable = st.sidebar.checkbox("Negotiable", value=True)

# Gabungkan options jadi string supaya sesuai pipeline
options_str = ",".join(options)

# ======================
# Buat DataFrame dari input
# ======================
data = {
    'Type':[type_],
    'Region':[region],
    'Make':[make],
    'Gear_Type':[gear_type],
    'Origin':[origin],
    'Options':[options_str],
    'Year':[year],
    'Engine_Size':[engine_size],
    'Mileage':[mileage],
    'Negotiable':[negotiable]
}
df = pd.DataFrame(data)

st.subheader("📋 Data yang Dimasukkan")
st.write(df)

# ======================
# Prediksi harga
# ======================
if st.button("Prediksi Harga"):
    try:
        prediction = model.predict(df)
        st.success(f"💰 Prediksi Harga Mobil Bekas: {prediction[0]:,.0f} Riyal")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")


## python -m streamlit run app_saudi_used_car.py (copy this to the terminal)

