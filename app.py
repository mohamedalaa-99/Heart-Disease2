import streamlit as st
import pandas as pd
import joblib

from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    return joblib.load("heart_model_clean.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction App")

Age = st.number_input("Age", 1, 120, 40)
RestingBP = st.number_input("RestingBP", 50, 200, 120)
Cholesterol = st.number_input("Cholesterol", 50, 600, 200)
MaxHR = st.number_input("MaxHR", 50, 250, 150)
Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

Sex = st.selectbox("Sex", ["M", "F"])
FastingBS = st.selectbox("FastingBS", [0, 1])
ExerciseAngina = st.selectbox("ExerciseAngina", ["Y", "N"])

ChestPainType = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
RestingECG = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
ST_Slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    data = pd.DataFrame({
        "Age":[Age],
        "RestingBP":[RestingBP],
        "Cholesterol":[Cholesterol],
        "MaxHR":[MaxHR],
        "Oldpeak":[Oldpeak],
        "Sex":[Sex],
        "FastingBS":[FastingBS],
        "ExerciseAngina":[ExerciseAngina],
        "ChestPainType":[ChestPainType],
        "RestingECG":[RestingECG],
        "ST_Slope":[ST_Slope]
    })

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"❌ احتمالية الإصابة: {prob*100:.2f}%")
    else:
        st.success(f"✔ سليم بنسبة: {(1-prob)*100:.2f}%")
