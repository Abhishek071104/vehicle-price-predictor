import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import qrcode
from io import BytesIO

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

def encode_input(value, col_name):
    encoder = encoders.get(col_name)
    if encoder and value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0

# Streamlit Config
st.set_page_config(page_title="Vehicle Price Predictor", page_icon="🚗", layout="centered")

# Sidebar
with st.sidebar:
    st.markdown("### 👤 Made by [M ABHISHEK](https://github.com/Abhishek071104)")
    st.markdown("🔗[GitHub](https://github.com/Abhishek071104)")
    st.markdown("🔗[LinkedIn](https://www.linkedin.com/in/-mabhishek/)")
    st.markdown("📧 [manipatruniabhishek07@gmail.com](mailto:manipatruniabhishek07@gmail.com)")
    st.markdown("---")
    st.markdown("### 📘 About This App")
    st.write("This application uses a trained **XGBoost model** to predict the estimated market price of a vehicle based on its specifications and condition. Built with XGBoost and Streamlit.")

    # QR Code (in-memory)
    qr = qrcode.make("https://vehiclepricepredictor.streamlit.app")
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    buffer.seek(0)
    st.image(buffer, caption="Scan to open app", use_container_width=True)

st.title("🚗 Vehicle Price Predictor")
st.markdown("Use the form below to get your vehicle's **estimated resale price**.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Layout using Tabs
tab1, tab2 = st.tabs(["🔮 Predict", "📜 History"])

with tab1:
    st.subheader("🔧 Basic Vehicle Details")

    col1, col2 = st.columns(2)
    with col1:
        make = st.text_input("🚘 Make", placeholder="e.g., Toyota")
        model_name = st.text_input("📌 Model", placeholder="e.g., Camry")
        year = st.number_input("📅 Year", min_value=1990, max_value=2025, value=2018)

    with col2:
        engine = st.text_input("🛠️ Engine", placeholder="e.g., 2.0L I4")
        cylinders = st.number_input("🔩 Cylinders", min_value=2, max_value=16, value=4)
        mileage = st.number_input("📍 Mileage (miles)", min_value=0, value=30000)

    st.subheader("⚙️ Specifications")

    col3, col4 = st.columns(2)
    with col3:
        fuel = st.selectbox("⛽ Fuel Type", ["Gasoline", "Diesel", "Electric", "Hybrid"])
        transmission = st.selectbox("🔁 Transmission", ["Automatic", "Manual", "CVT"])
        drivetrain = st.selectbox("🚙 Drivetrain", ["Front-wheel Drive", "Rear-wheel Drive", "All-wheel Drive", "Four-wheel Drive"])

    with col4:
        body = st.selectbox("🚘 Body Style", ["Sedan", "SUV", "Pickup Truck", "Coupe", "Hatchback", "Van", "Other"])
        doors = st.number_input("🚪 Doors", min_value=2, max_value=6, value=4)
        trim = st.text_input("🎯 Trim", value="Unknown")

    st.subheader("🎨 Appearance")
    col5, col6 = st.columns(2)
    with col5:
        exterior_color = st.text_input("🌈 Exterior Color", value="White")
    with col6:
        interior_color = st.text_input("🛋️ Interior Color", value="Black")

    # Vehicle image preview
    if make and model_name:
        st.image(f"https://source.unsplash.com/400x200/?{make},{model_name}", caption="Sample Image", use_container_width=False)

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
        # Show progress bar while "loading"
        progress_text = "Predicting vehicle price..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        price = int(prediction)

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        st.success(f"💵 Estimated Price: **${int(prediction):,}**")

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

with tab2:
    if st.session_state.history:
        st.markdown("### 📊 Previous Predictions")
        df_history = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(df_history, use_container_width=True)

        csv = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "vehicle_price_history.csv", "text/csv")

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
# Banner
st.image("360_F_910998153_tOayMd30RZjpx2kzh9baGdcLBDXwMj00.jpg", use_container_width=True)
