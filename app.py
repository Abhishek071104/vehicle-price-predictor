import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚘",
    layout="wide"
)

# -------------------- Load Lottie Animation --------------------
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# Working Lottie animation (moving car)
lottie_car = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_u4yrau.json")

# -------------------- Load Model & Data --------------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost_vehicle_price_model.pkl")

model = load_model()
df_sample = pd.read_csv("dataset.csv")
categorical_cols = df_sample.select_dtypes(include=["object"]).columns.tolist()

def load_label_encoders(df, cat_cols):
    return {
        col: {label: idx for idx, label in enumerate(df[col].astype(str).unique())}
        for col in cat_cols
    }

encoders = load_label_encoders(df_sample, categorical_cols)

def encode_input(val, col_name):
    return encoders.get(col_name, {}).get(val, 0)

# -------------------- Session State --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("8f0ee631-2a1f-4b9a-b296-d0165ab82a84.png", width=100)
    st.title("📌 About")
    st.markdown("""
    🚗 **Vehicle Price Predictor**  
    Built with **XGBoost + Streamlit**

    ---

    👨‍💻 Made by [Abhishek Manipatruni](https://www.linkedin.com/in/-mabhishek/)

    🐙 [GitHub](https://github.com/Abhishek071104)  
    💼 [LinkedIn](https://www.linkedin.com/in/-mabhishek/)  
    📧 [Email](https://mail.google.com/mail/?view=cm&to=manipatruniabhishek07@gmail.com)
    """)

# -------------------- Header --------------------
col1, col2 = st.columns([1.5, 1])
with col1:
    st.title("🚘 Vehicle Price Predictor")
    st.markdown("Enter vehicle details to get an estimated market price.")
with col2:
    if lottie_car:
        st_lottie(lottie_car, height=150, key="car")
    else:
        st.image("8f0ee631-2a1f-4b9a-b296-d0165ab82a84.png", width=150)

st.markdown("---")

# -------------------- Input Form --------------------
st.subheader("🧾 Enter Vehicle Details")
col1, col2 = st.columns(2)
with col1:
    make = st.text_input("Make", "Toyota")
    model_input = st.text_input("Model", "Camry")
    year = st.number_input("Year", 1990, 2025, 2019)
    engine = st.selectbox("Engine", df_sample["engine"].dropna().unique())
    cylinders = st.selectbox("Cylinders", sorted(df_sample["cylinders"].dropna().unique()))
    transmission = st.selectbox("Transmission", df_sample["transmission"].dropna().unique())
    trim = st.selectbox("Trim", df_sample["trim"].dropna().unique())
with col2:
    fuel = st.selectbox("Fuel Type", df_sample["fuel"].dropna().unique())
    mileage = st.number_input("Mileage (in miles)", 0, 500000, 30000)
    body = st.selectbox("Body Type", df_sample["body"].dropna().unique())
    doors = st.selectbox("Doors", sorted(df_sample["doors"].dropna().unique()))
    exterior_color = st.selectbox("Exterior Color", df_sample["exterior_color"].dropna().unique())
    interior_color = st.selectbox("Interior Color", df_sample["interior_color"].dropna().unique())
    drivetrain = st.selectbox("Drivetrain", df_sample["drivetrain"].dropna().unique())

# -------------------- Prediction --------------------
if st.button("🔍 Predict Price"):
    input_dict = {
        "make": encode_input(make, "make"),
        "model": encode_input(model_input, "model"),
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
    pred = model.predict(input_df)[0]
    st.success(f"💵 **Estimated Price: ₹{int(pred):,}**")
    st.session_state.history.append({
        "Make": make,
        "Model": model_input,
        "Year": year,
        "Mileage": mileage,
        "Price": int(pred)
    })

# -------------------- History --------------------
if st.session_state.history:
    st.markdown("### 🕓 Previous Predictions")
    st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------- Bar Chart --------------------
st.markdown("### 📊 Example: Mileage vs Price Trend")
chart_df = pd.DataFrame({
    'Mileage': [0, 20000, 40000, 60000, 80000],
    'Predicted Price': [45000, 40000, 35000, 30000, 25000]
})
st.bar_chart(chart_df.set_index("Mileage"))
