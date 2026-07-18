
import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load trained model
# ===========================
model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Predictor")
st.write("Enter the car details below to estimate the price.")

# ===========================
# User Inputs
# ===========================

brand = st.selectbox(
    "Brand",
    ["Tesla", "BMW", "Audi", "Ford", "Honda", "Mercedes", "Toyota"]
)

model_name = st.selectbox(
    "Model",
    [
        "Model S", "Model 3", "Model X", "Model Y",
        "3 Series", "5 Series", "X3", "X5",
        "A3", "A4", "Q5", "Q7",
        "Mustang", "Explorer", "Fiesta", "Focus",
        "Civic", "Accord", "CR-V", "Fit",
        "C-Class", "E-Class", "GLA", "GLC",
        "Camry", "Corolla", "Prius", "RAV4"
    ]
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2020
)

engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.5,
    max_value=8.0,
    value=2.0,
    step=0.1
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Electric", "Diesel", "Hybrid"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    value=50000,
    step=1000
)

condition = st.selectbox(
    "Condition",
    ["New", "Used", "Like New"]
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Brand": [brand],
        "Year": [year],
        "Engine Size": [engine_size],
        "Fuel Type": [fuel_type],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Condition": [condition],
        "Model": [model_name]
    })

    # One-Hot Encode
    input_df = pd.get_dummies(input_df)

    # Match training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")

st.markdown("---")
st.caption("Developed by Shumaila jawed | Machine Learning Project")
