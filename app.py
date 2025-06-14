import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import requests

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="https://cdn-icons-png.flaticon.com/512/743/743007.png",
    layout="wide"
)

# -------------------- Load Lottie Animation --------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_car = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_pprxh53t.json")

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
    st.image("https://cdn-icons-png.flaticon.com/512/743/743007.png", width=80)
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
col1, col2 = st.columns([1, 2])
with col1:
    st.title("Vehicle Price Predictor")
    st.markdown("Enter vehicle details to get an estimated market price.")
with col2:
    st_lottie(lottie_car, height=180, key="car")

st.markdown("---")

# -------------------- Input Form --------------------
st.subheader("🧾 Enter Vehicle Details")
col1, col2 = st.columns(2)
with col1:
    make = st.text_input("Make", "Toyota")
    model_input = st.text_input("Model", "Camry")
    year = st.number_input("Year", 1990, 2025, 2019)
    transmission = st.selectbox("Transmission", df_sample["transmission"].dropna().unique())
    fuel = st.selectbox("Fuel Type", df_sample["fuel"].dropna().unique())
    engine = st.selectbox("Engine", df_sample["engine"].dropna().unique())
    cylinders = st.selectbox("Cylinders", sorted(df_sample["cylinders"].dropna().unique()))
    trim = st.selectbox("Trim", df_sample["trim"].dropna().unique())
with col2:
    mileage = st.number_input("Mileage (in km)", 0, 500000, 35000)
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
        "transmission": encode_input(transmission, "transmission"),
        "fuel": encode_input(fuel, "fuel"),
        "engine": encode_input(engine, "engine"),
        "cylinders": cylinders,
        "trim": encode_input(trim, "trim"),
        "mileage": mileage,
        "body": encode_input(body, "body"),
        "doors": doors,
        "exterior_color": encode_input(exterior_color, "exterior_color"),
        "interior_color": encode_input(interior_color, "interior_color"),
        "drivetrain": encode_input(drivetrain, "drivetrain")
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    st.success(f"💵 **Estimated Price: ₹{int(prediction):,}**")

    st.session_state.history.append({
        "Make": make,
        "Model": model_input,
        "Year": year,
        "Mileage": mileage,
        "Predicted Price": int(prediction)
    })

# -------------------- History --------------------
if st.session_state.history:
    st.markdown("### 🕓 Previous Predictions")
    st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------- Bar Chart --------------------
st.markdown("### 📊 Example: Mileage vs Price Trend")
chart_data = pd.DataFrame({
    'Mileage': [0, 20000, 40000, 60000, 80000],
    'Predicted Price': [45000, 40000, 35000, 30000, 25000]
})
st.bar_chart(chart_data.set_index("Mileage"))
