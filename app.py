import streamlit as st
import pandas as pd
import joblib
import pickle
import sys
import importlib

# FORCE IMPORT ALL NEEDED CLASSES
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# FIX PICKLE ENVIRONMENT
class Dummy:
    pass

modules = [
    "category_encoders.binary",
    "sklearn.preprocessing",
    "sklearn.compose",
    "sklearn.decomposition",
    "sklearn.pipeline",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.svm"
]

for m in modules:
    try:
        importlib.import_module(m)
    except:
        pass


@st.cache_resource
def load_model():
    with open("heart_disease_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
