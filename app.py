import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# -------------------- Load Model & Encoders --------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_vehicle_price_model.pkl")

model = load_model()

# -------------------- Lottie Animation --------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_car = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json")

# -------------------- UI --------------------
st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📌 About")
    st.markdown("""
    🚗 **Vehicle Price Predictor**  
    Built with **XGBoost + Streamlit**  
    Predict accurate market prices based on:
    - Brand, Model, Year  
    - Transmission, Fuel type  
    - Mileage & More  
    ---
    👨‍💻 Developed by [Abhishek Manipatruni](#)
    """)

# Header
col1, col2 = st.columns([1, 2])
with col1:
    st.title("🚗 Vehicle Price Predictor")
    st.markdown("Enter vehicle details to get an estimated market price.")
with col2:
    st_lottie(lottie_car, height=180, key="car")

st.markdown("---")

# Sample data for encoding (you can improve this by saving your encoders)
df_sample = pd.read_csv("dataset.csv")
categorical_cols = df_sample.select_dtypes(include=["object"]).columns.tolist()

def load_label_encoders(df, cat_cols):
    encoders = {}
    for col in cat_cols:
        encoders[col] = {label: idx for idx, label in enumerate(df[col].astype(str).unique())}
    return encoders

encoders = load_label_encoders(df_sample, categorical_cols)

def encode_input(val, col_name):
    return encoders.get(col_name, {}).get(val, 0)

# -------------------- Input Section --------------------
st.subheader("🧾 Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    make = st.text_input("Make", "Toyota")
    model_input = st.text_input("Model", "Camry")
    year = st.number_input("Year", 1990, 2025, 2019)
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])

with col2:
    mileage = st.number_input("Mileage (in km)", 0, 500000, 35000)
    engine = st.selectbox("Engine Size", ["1.2L", "1.5L", "2.0L", "3.0L", "Electric"])
    body = st.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Coupe"])
    doors = st.selectbox("Doors", [2, 3, 4, 5])

# -------------------- Predict Button --------------------
if st.button("🔍 Predict Price"):
    input_dict = {
        "make": encode_input(make, "make"),
        "model": encode_input(model_input, "model"),
        "year": year,
        "transmission": encode_input(transmission, "transmission"),
        "fuel": encode_input(fuel, "fuel"),
        "mileage": mileage,
        "engine": encode_input(engine, "engine"),
        "body_type": encode_input(body, "body_type"),
        "doors": doors,
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    st.success(f"💵 **Estimated Price: ₹{int(prediction):,}**")

    st.markdown("---")
    st.markdown("📈 *Prediction is based on past sales data using XGBoost Regression.*")
