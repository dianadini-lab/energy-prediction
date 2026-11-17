import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Energy Consumption Prediction",
    page_icon="⚡",
    layout="wide"
)

# 2. Judul Utama (Nuansa Hijau-Putih)
st.markdown(
    "<h1 style='color: #4B6B32;'>HASIL PREDIKSI <span style='font-size: 20px; color: #555;'>(Linear Regression Model)</span></h1>", 
    unsafe_allow_html=True
)

# 3. Fungsi Memuat Model .pkl
@st.cache_resource
def load_model():
    return joblib.load("model_linear_regression.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat file 'model_linear_regression.pkl'. Pastikan file berada satu folder dengan file python ini. Detail error: {e}")
    st.stop()

# 4. Layout Dua Kolom Utama
col_left, col_right = st.columns([1, 1], gap="large")

# --- KOLOM KIRI: INPUT PREDIKTOR ---
with col_left:
    st.markdown("<h3 style='color: #4B6B32;'>USER MENGINPUT PREDIKTOR</h3>", unsafe_allow_html=True)
    st.markdown("**Energy Consumption Prediction**")
    st.write("Silahkan masukkan parameter gedung dan operasional ..")
    
    
    # Widget Input Numerik (Slider)
    temperature = st.slider("Temperature (°C):", 20.0, 30.0, 21.0, 0.1)
    humidity = st.slider("Humidity (%):", 30.0, 60.0, 38.4, 0.1)
    square_footage = st.slider("Square Footage:", 1000.0, 2000.0, 1500.0, 1.0)
    occupancy = st.slider("Occupancy (People):", 0, 9, 7, 1)
    renewable_energy = st.slider("Renewable Energy (kWh):", 0.0, 30.0, 10.0, 0.1)
    
    # Widget Input Kategorikal (Dropdown)
    hvac_usage = st.selectbox("HVAC Usage:", ["Off", "On"])
    lighting_usage = st.selectbox("Lighting Usage:", ["Off", "On"], index=1)
    holiday = st.selectbox("Holiday:", ["No", "Yes"])
    
    # Widget Input Waktu
    hour = st.slider("Hour:", 0, 23, 18, 1)
    day_type = st.selectbox("Day Type:", ["Weekend", "Weekday"])

# --- KOLOM KANAN: OUTPUT PREDIKSI & GRAFIK ---
with col_right:
    st.markdown("<h3 style='color: #4B6B32;'>HASIL PREDIKSI</h3>", unsafe_allow_html=True)
    
    # Tombol Aksi
    predict_button = st.button("Predict Energy Consumption", use_container_width=True, type="primary")
    
    if predict_button:
        # Menyiapkan DataFrame untuk prediksi tunggal sesuai input user
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'SquareFootage': [square_footage],
            'Occupancy': [occupancy],
            'RenewableEnergy': [renewable_energy],
            'HVACUsage': [hvac_usage],
            'LightingUsage': [lighting_usage],
            'Holiday': [holiday],
            'Hour': [hour]
        })
        
        # Eksekusi Prediksi Utama
        predicted_energy = model.predict(input_data)[0]
        
        # Tampilan Ringkasan Teks Monospace
        result_text = f"""
===============================================
         ENERGY CONSUMPTION PREDICTION         
===============================================
Predicted Energy Consumption : {predicted_energy:.2f} kwh
Period                       : 1 hour
-----------------------------------------------
Input Summary:
Temperature                  : {temperature}
Humidity                     : {humidity}
Square Footage               : {int(square_footage)}
Occupancy                    : {occupancy}
HVAC Usage                   : {hvac_usage}
Lighting Usage               : {lighting_usage}
Renewable Energy             : {renewable_energy}
Holiday                      : {holiday}
Hour                         : {hour:02d}:00
Day Type                     : {day_type}
"""
        st.code(result_text, language="text")
        
        # --- VISUALISASI GRAFIK SIMULASI 24 JAM ---
        st.markdown("#### Simulasi Tren Konsumsi Energi (24 Jam)")
        st.caption("Grafik ini menunjukkan estimasi perubahan konsumsi energi dari jam 00:00 hingga 23:00 jika kondisi lainnya tetap sama.")
        
        # Membuat data simulasi dari jam 0 sampai 23
        hours_range = list(range(24))
        sim_data = pd.DataFrame({
            'Temperature': [temperature] * 24,
            'Humidity': [humidity] * 24,
            'SquareFootage': [square_footage] * 24,
            'Occupancy': [occupancy] * 24,
            'RenewableEnergy': [renewable_energy] * 24,
            'HVACUsage': [hvac_usage] * 24,
            'LightingUsage': [lighting_usage] * 24,
            'Holiday': [holiday] * 24,
            'Hour': hours_range
        })
        
        # Prediksi untuk seluruh 24 jam
        sim_predictions = model.predict(sim_data)
        
        # Plotting menggunakan Matplotlib
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(hours_range, sim_predictions, color='#4B6B32', linewidth=2.5, marker='o', markersize=4, label='Tren Prediksi 24 Jam')
        ax.scatter([hour], [predicted_energy], color='#D9534F', s=100, zorder=5, label=f'Pilihanmu ({hour:02d}:00 = {predicted_energy:.2f} kWh)')
        ax.axvline(x=hour, color='#D9534F', linestyle='--', alpha=0.5)
        
        ax.set_title("Estimasi Konsumsi Energi Sepanjang Hari", fontsize=11, fontweight='bold', color='#333')
        ax.set_xlabel("Jam (00:00 - 23:00)", fontsize=9)
        ax.set_ylabel("Konsumsi Energi (kWh)", fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8)
        
        st.pyplot(fig)
    else:
        st.info("👈 Atur nilai prediktor di sebelah kiri, lalu klik tombol **Predict Energy Consumption** untuk melihat hasil prediksi dan grafiknya.")
