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
        return 0  # fallback for unseen values

# ---- Streamlit UI Starts Here ----

st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Vehicle Price Predictor")
st.markdown("Enter vehicle details below to get an estimated **market price** using a machine learning model.")

st.divider()
st.subheader("🔧 Basic Vehicle Details")

col1, col2 = st.columns(2)
with col1:
    make = st.text_input("Make", placeholder="e.g., Toyota", help="Manufacturer of the vehicle")
    model_name = st.text_input("Model", placeholder="e.g., Corolla", help="Specific model of the vehicle")
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2018, help="Manufacturing year")

with col2:
    engine = st.text_input("Engine", placeholder="e.g., 2.0L I4", help="Engine type or size")
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, help="Number of cylinders")
    mileage = st.number_input("Mileage (miles)", min_value=0, value=30000, help="Total distance driven")

st.subheader("⚙️ Specifications")

col3, col4 = st.columns(2)
with col3:
    fuel = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
    drivetrain = st.selectbox("Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive"])

with col4:
    body = st.selectbox("Body Style", ["Sedan", "SUV", "Pickup Truck", "Coupe", "Hatchback", "Van", "Other"])
    doors = st.number_input("Doors", min_value=2, max_value=6, value=4)
    trim = st.text_input("Trim", value="Unknown", help="Trim level (e.g., XLE, SE)")

st.subheader("🎨 Appearance")

col5, col6 = st.columns(2)
with col5:
    exterior_color = st.text_input("Exterior Color", value="White")
with col6:
    interior_color = st.text_input("Interior Color", value="Black")

st.markdown("---")

# Prediction logic
if st.button("🎯 Predict Price"):
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
    
    st.success(f"💵 **Estimated Price: ${int(prediction):,}**")
    st.balloons()
  
# Store prediction history in session
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🎯 Predict Price"):
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
    
    st.success(f"💵 **Estimated Price: ${int(prediction):,}**")
    st.balloons()

    # Store display version (unencoded for clarity)
    display_data = {
        "Make": make,
        "Model": model_name,
        "Year": year,
        "Mileage": mileage,
        "Engine": engine,
        "Fuel": fuel,
        "Transmission": transmission,
        "Body": body,
        "Price ($)": int(prediction)
    }
    st.session_state.history.append(display_data)

# Show history if exists
if st.session_state.history:
    st.markdown("## 📜 Previous Predictions")
    st.dataframe(pd.DataFrame(st.session_state.history).iloc[::-1], use_container_width=True)

