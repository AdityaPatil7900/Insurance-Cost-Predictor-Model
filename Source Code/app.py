import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------- Load Model and Scaler -----------
@st.cache_resource
def load_model():
    model = joblib.load("insurance_model.pkl")   # Ensure file exists
    scaler = joblib.load("scaler.pkl")          # Ensure file exists
    return model, scaler

model, scaler = load_model()

# ----------- Streamlit Page Config -----------
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="centered")

# ----------- Custom CSS for Styling -----------
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #38B82F6;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }
    .predict-btn {
        background-color: #2563eb;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Title Section -----------
st.markdown('<div class="title">💰 Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate your medical insurance cost based on your personal details</div>', unsafe_allow_html=True)

# ----------- Input Form -----------
st.subheader("Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ----------- Prepare Input Data -----------
input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# ----------- Prediction Button -----------
if st.button("📊 Predict Insurance Cost", use_container_width=True):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Insurance Cost: **${prediction:,.2f}**")

# ----------- Footer -----------
st.markdown("""
---
<div style='text-align:center; font-size:14px; color:#4b5563'>
Developed by <b>Aditya Patil</b> | <a href='https://github.com/AdityaPatil7900' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
