import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("xgboost_vehicle_price_model.pkl")

# Load sample data for label encoding
df_sample = pd.read_csv("dataset.csv")
df_sample.fillna("Unknown", inplace=True)

# Define categorical columns
categorical_cols = ['make', 'model', 'engine', 'fuel', 'transmission', 'trim',
                    'body', 'exterior_color', 'interior_color', 'drivetrain']

# Build consistent label encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_sample[col] = le.fit_transform(df_sample[col])
    encoders[col] = le

# Encoding function
def encode_input(value, col_name):
    encoder = encoders.get(col_name)
    if encoder and value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0  # fallback if unseen

# ---- Streamlit UI Starts Here ----

st.title("🚗 Vehicle Price Predictor")
st.write("Enter vehicle details to get an estimated market price.")

# Input fields
make = st.text_input("Make (e.g., Toyota, Ford, BMW)")
model_name = st.text_input("Model (e.g., Camry, Mustang)")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
engine = st.text_input("Engine (e.g., 2.0L I4)")
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4)
fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
mileage = st.number_input("Mileage (miles)", min_value=0, value=30000)
transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
trim = st.text_input("Trim", value="Unknown")
body = st.selectbox("Body Style", ["Sedan", "SUV", "Pickup Truck", "Coupe", "Hatchback", "Van", "Other"])
doors = st.number_input("Doors", min_value=2, max_value=6, value=4)
exterior_color = st.text_input("Exterior Color", value="White")
interior_color = st.text_input("Interior Color", value="Black")
drivetrain = st.selectbox("Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive"])

# Prediction button
if st.button("Predict Price"):
    input_dict = {
        "make": encode_input(make, "make"),
        "model": encode_input(model_name, "model"),
        "year": year,
        "engine": encode_input(engine, "engine"),
        "cylinders": cylinders,
        "fuel": encode_input(fuel, "fuel"),
        "mileage": mileage,
        "transmission": encode_input(transmission, "transmission"),
        "trim": encode_input(trim, "trim"),
        "body": encode_input(body, "body"),
        "doors": doors,
        "exterior_color": encode_input(exterior_color, "exterior_color"),
        "interior_color": encode_input(interior_color, "interior_color"),
        "drivetrain": encode_input(drivetrain, "drivetrain")
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    st.success(f"💵 Estimated Price: ${int(prediction):,}")
