import streamlit as st
import pandas as pd
import joblib

# -------------------------
# IMPORT ALL OBJECTS USED IN PIPELINE
# -------------------------

# Encoders
from category_encoders import BinaryEncoder

# Sklearn Preprocessing
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# PCA
from sklearn.decomposition import PCA

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Pipeline
from sklearn.pipeline import Pipeline

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model_pipeline.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction App")
st.write("تطبيق للتنبؤ بمرض القلب باستخدام نموذج Random Forest Pipeline.")

# -----------------------------
# Input Fields
# -----------------------------
Age = st.number_input("Age", 1, 120, 45)
RestingBP = st.number_input("Resting Blood Pressure", 50, 250, 120)
Cholesterol = st.number_input("Cholesterol", 50, 600, 200)
MaxHR = st.number_input("Max Heart Rate", 50, 250, 150)
Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

Sex = st.selectbox("Sex", ["M", "F"])
FastingBS = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])

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

    if prediction == 1:
        st.error(f"❌ احتمالية الإصابة: {probability*100:.2f}%")
    else:
        st.success(f"✔ المريض سليم بنسبة: {(1-probability)*100:.2f}%")
