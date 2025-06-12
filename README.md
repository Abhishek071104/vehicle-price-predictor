# vehicle-price-predictor
Vehicle Price Predictor – ML + Streamlit : An end-to-end machine learning project that predicts used car prices based on specifications like make, model, year, mileage, engine type, and more. Built with XGBoost, Pandas, Scikit-learn, and deployed using Streamlit for an interactive web experience. 

# Problem Statement
Accurately estimate a vehicle’s market value using its key attributes — helping car buyers, sellers, and dealerships make informed decisions.

# Technologies Used
-Python 3,
-XGBoost Regressor (for accurate predictions),
-Scikit-learn (model evaluation & preprocessing),
-Streamlit (for a clean, interactive UI),
-Pandas, NumPy (data handling)

# Project Structure
vehicle-price-predictor
-app.py                  # Streamlit web app,
-dataset.csv             # Vehicle dataset with specs and prices,
-xgboost_vehicle_price_model.pkl  # Trained model (saved with joblib),
-requirements.txt        # Python dependencies,
-README.md               # Project documentation

Note: `xgboost_vehicle_price_model.pkl` is a serialized binary ML model and cannot be viewed directly on GitHub.
It is loaded at runtime by the Streamlit app.
