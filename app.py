import streamlit as st
import pandas as pd
import joblib

# Required for loading BinaryEncoder inside the pipeline
from category_encoders import BinaryEncoder

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model_pipeline.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction App")
st.write("تطبيق للتنبؤ بمرض القلب باستخدام النموذج الذي قمت بتدريبه.")

# -----------------------------
# Input Form
# -----------------------------
st.subheader("📝 أدخل بيانات المريض")

Age = st.number_input("Age", min_value=1, max_value=120, value=45)
RestingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
Cholesterol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
MaxHR = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)

Sex = st.selectbox("Sex", ["M", "F"])
FastingBS = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", [0, 1])
ExerciseAngina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict"):
    input_data = pd.DataFrame({
        "Age": [Age],
        "RestingBP": [RestingBP],
        "Cholesterol": [Cholesterol],
        "MaxHR": [MaxHR],
        "Oldpeak": [Oldpeak],
        "Sex": [Sex],
        "FastingBS": [FastingBS],
        "ExerciseAngina": [ExerciseAngina],
        "ChestPainType": [ChestPainType],
        "RestingECG": [RestingECG],
        "ST_Slope": [ST_Slope]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 النتيجة:")
    if prediction == 1:
        st.error(f"❌ المريض لديه احتمالية إصابة بمرض القلب بنسبة: {probability*100:.2f}%")
    else:
        st.success(f"✔ المريض سليم بنسبة: {(1 - probability)*100:.2f}%")
