import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load Model and Feature Names
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
st.write("Fill in the details below to estimate the car price.")

# ===========================
# User Inputs
# ===========================

brand = st.selectbox(
    "Brand",
    [
        "Tesla",
        "BMW",
        "Audi",
        "Ford",
        "Honda",
        "Mercedes",
        "Toyota"
    ]
)

model_name = st.selectbox(
    "Model",
    [
        "Civic",
        "CR-V",
        "Corolla",
        "E-Class",
        "Fiesta",
        "Fit",
        "Focus",
        "GLA",
        "GLC",
        "Model 3",
        "Model X",
        "Model Y",
        "Mustang",
        "Prius",
        "Q5",
        "Q7",
        "RAV4",
        "X3",
        "X5"
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
    [
        "Petrol",
        "Electric",
        "Diesel",
        "Hybrid"
    ]
)

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=500000,
    value=50000,
    step=1000
)

condition = st.selectbox(
    "Condition",
    [
        "New",
        "Used",
        "Like New"
    ]
)

# ===========================
# Prediction
# ===========================

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Brand": [brand],
        "Year": [year],
        "Engine Size": [engine_size],
        "Fuel Type": [fuel_type],
        "Transmission": [transmission],
        "Mileage": [mileage],
        "Condition": [condition],
        "Model": [model_name]
    })

    # One-Hot Encode User Input
    input_data = pd.get_dummies(input_data)

    # Match Training Features
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")

st.markdown("---")
st.caption("Developed by Shumaila Javed | Machine Learning Project")
