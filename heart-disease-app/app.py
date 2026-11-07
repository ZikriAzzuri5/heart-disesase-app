import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Prediksi Risiko Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #e63946;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border-left: 5px solid #e63946;
    }
    .result-box-positive {
        background-color: #fff5f5;
        border: 2px solid #feb2b2;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-box-negative {
        background-color: #f0fff4;
        border: 2px solid #9ae6b4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and scaler
@st.cache_resource
def load_assets():
    model_path = "heart_disease_model.pkl"
    scaler_path = "heart_disease_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("File model atau scaler tidak ditemukan!")
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# Header
st.markdown("<div class='main-header'>❤️ Aplikasi Prediksi Risiko Penyakit Jantung</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Masukan data medis dan gaya hidup pasien untuk memprediksi tingkat risiko penyakit jantung secara ilmiah.</div>", unsafe_allow_html=True)

if model is not None and scaler is not None:
    # Form input with two columns
    st.markdown("### 📋 Form Input Data Pasien")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Informasi Demografi & Gaya Hidup")
        
        age = st.number_input("Umur (Tahun)", min_value=18, max_value=120, value=50, step=1)
        sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
        smoker_status = st.selectbox("Status Merokok", ["Tidak Pernah (Never)", "Mantan Perokok (Former)", "Perokok Aktif (Current)"])
        alcohol = st.number_input("Konsumsi Alkohol (Unit/Minggu)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        
        exercise_mins = st.number_input("Olahraga (Menit/Minggu)", min_value=0, max_value=1000, value=150, step=10)
        daily_steps = st.number_input("Langkah Harian", min_value=0, max_value=50000, value=6000, step=500)
        sleep_hours = st.number_input("Jam Tidur per Hari", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
        
        stress_score = st.slider("Skor Stres (0 - 100)", min_value=0, max_value=100, value=45)
        diet_score = st.slider("Skor Kualitas Diet (0 - 100)", min_value=0, max_value=100, value=60)
        
        wearable_owner = st.radio("Menggunakan Smartwatch / Health Tracker?", ["Tidak", "Ya"], horizontal=True)
        family_history = st.radio("Ada Riwayat Keluarga Penyakit Jantung?", ["Tidak", "Ya"], horizontal=True)

    with col2:
        st.subheader("🩺 Pemeriksaan Klinis & Laboratorium")
        
        bp_sys = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=70, max_value=250, value=125, step=1)
        bp_dia = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=40, max_value=150, value=80, step=1)
        
        chol_total = st.number_input("Kolesterol Total (mg/dL)", min_value=80, max_value=600, value=190, step=1)
        hdl = st.number_input("Kolesterol HDL (mg/dL)", min_value=10, max_value=150, value=55, step=1)
        ldl = st.number_input("Kolesterol LDL (mg/dL)", min_value=20, max_value=400, value=100, step=1)
        triglycerides = st.number_input("Trigliserida (mg/dL)", min_value=30, max_value=1000, value=150, step=1)
        
        fbs = st.number_input("Gula Darah Puasa (mg/dL)", min_value=50, max_value=500, value=115, step=1)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=18.0, value=5.7, step=0.1)
        
        resting_hr = st.number_input("Detak Jantung Istirahat (bpm)", min_value=40, max_value=200, value=80, step=1)
        max_hr = st.number_input("Detak Jantung Maksimum Dicapai (bpm)", min_value=60, max_value=230, value=165, step=1)
        
        chest_pain = st.selectbox(
            "Tipe Nyeri Dada",
            ["Tidak Ada (Asymptomatic)", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"]
        )
        
        exercise_angina = st.radio("Mengalami Nyeri Dada Saat Olahraga?", ["Tidak", "Ya"], horizontal=True)
        st_depression = st.number_input("ST Depression (mm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.markdown("---")
    
    # Predict button
    predict_btn = st.button("🔍 Analisis Risiko Sekarang", type="primary", use_container_width=True)
    
    if predict_btn:
        # Preprocessing input values to match exact feature_names_in_ expected by scaler
        cp_atypical = 1 if chest_pain == "Atypical Angina" else 0
        cp_nonanginal = 1 if chest_pain == "Non-Anginal Pain" else 0
        cp_typical = 1 if chest_pain == "Typical Angina" else 0
        
        smoker_former = 1 if smoker_status == "Mantan Perokok (Former)" else 0
        smoker_never = 1 if smoker_status == "Tidak Pernah (Never)" else 0
        
        sex_male = 1 if sex == "Laki-laki" else 0
        ex_angina_val = 1 if exercise_angina == "Ya" else 0
        fam_history_val = 1 if family_history == "Ya" else 0
        wearable_val = 1 if wearable_owner == "Ya" else 0
        
        # Create input DataFrame with exact column names in order
        input_data = pd.DataFrame([{
            'age': age,
            'resting_bp_systolic': bp_sys,
            'resting_bp_diastolic': bp_dia,
            'cholesterol_total': chol_total,
            'hdl': hdl,
            'ldl': ldl,
            'triglycerides': triglycerides,
            'fasting_blood_sugar': fbs,
            'hba1c': hba1c,
            'bmi': bmi,
            'resting_heart_rate': resting_hr,
            'max_heart_rate_achieved': max_hr,
            'exercise_induced_angina': ex_angina_val,
            'st_depression': st_depression,
            'family_history': fam_history_val,
            'alcohol_units_per_week': alcohol,
            'exercise_minutes_per_week': exercise_mins,
            'sleep_hours': sleep_hours,
            'stress_score': stress_score,
            'wearable_owner': wearable_val,
            'daily_steps': daily_steps,
            'diet_quality_score': diet_score,
            'chest_pain_type_Atypical Angina': cp_atypical,
            'chest_pain_type_Non-Anginal Pain': cp_nonanginal,
            'chest_pain_type_Typical Angina': cp_typical,
            'smoker_status_Former': smoker_former,
            'smoker_status_Never': smoker_never,
            'sex_Male': sex_male
        }])
        
        # Scale features
        scaled_input = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            prob_risk = probabilities[1] * 100
        else:
            prob_risk = 100.0 if prediction == 1 else 0.0
            
        # Display Result
        st.markdown("## 📊 Hasil Analisis Prediksi")
        
        if prediction == 1:
            st.markdown(f"""
            <div class='result-box-positive'>
                <div class='result-title' style='color: #c53030;'>⚠️ TERINDIKASI RISIKO TINGGI PENYAKIT JANTUNG</div>
                <p style='font-size: 1.2rem; color: #742a2a;'>
                    Estimasi Probabilitas Risiko: <b>{prob_risk:.1f}%</b>
                </p>
                <p style='color: #4a5568;'>
                    Sistem mendeteksi pola data klinis dan gaya hidup yang mengarah pada peningkatan risiko masalah kardiovaskular. 
                    Disarankan untuk segera berkonsultasi dengan dokter spesialis jantung untuk pemeriksaan medis komprehensif.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-box-negative'>
                <div class='result-title' style='color: #276749;'>✅ RISIKO PENYAKIT JANTUNG RELATIF RENDAH</div>
                <p style='font-size: 1.2rem; color: #22543d;'>
                    Estimasi Probabilitas Risiko: <b>{prob_risk:.1f}%</b>
                </p>
                <p style='color: #4a5568;'>
                    Berdasarkan data yang dimasukkan, indikator medis dan gaya hidup berada dalam rentang risiko yang rendah. 
                    Tetap pertahankan pola hidup sehat, konsumsi makanan bergizi, dan olahraga secara teratur.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.progress(int(prob_risk))

st.sidebar.markdown("### ℹ️ Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini dibangun menggunakan model Machine Learning (**Logistic Regression**) "
    "dan **StandardScaler** yang telah dilatih pada data parameter kesehatan kardiovaskular.\n\n"
    "⚠️ *Catatan: Aplikasi ini hanya berfungsi sebagai alat bantu skrining awal dan tidak menggantikan diagnosis medis resmi oleh dokter.*"
)
