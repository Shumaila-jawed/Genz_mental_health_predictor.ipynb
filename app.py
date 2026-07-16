


import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("mental_health_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠")

st.title("Mental Health Score Predictor")
st.write("Predict the mental health score based on social media usage.")

age = st.number_input("Age", min_value=10, max_value=100, value=20)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])

country = st.text_input("Country", "Pakistan")

daily_usage_hours = st.number_input("Daily Usage Hours", 0.0, 24.0, 4.0)

primary_platform = st.selectbox(
    "Primary Platform",
    ["Instagram","Facebook","TikTok","YouTube","Twitter","Snapchat"]
)

num_platforms_used = st.number_input(
    "Number of Platforms Used",
    min_value=1,
    max_value=10,
    value=3
)

purpose = st.selectbox(
    "Purpose",
    ["Entertainment","Education","Communication","Business"]
)

avg_session_minutes = st.number_input(
    "Average Session Minutes",
    min_value=1,
    max_value=500,
    value=60
)

night_usage = st.selectbox(
    "Night Usage",
    ["Yes","No"]
)

addiction_level = st.selectbox(
    "Addiction Level",
    ["Low","Medium","High"]
)

screen_time_before_sleep = st.number_input(
    "Screen Time Before Sleep (Minutes)",
    min_value=0,
    max_value=300,
    value=30
)

if st.button("Predict"):

    data = pd.DataFrame({
        "age":[age],
        "gender":[gender],
        "country":[country],
        "daily_usage_hours":[daily_usage_hours],
        "primary_platform":[primary_platform],
        "num_platforms_used":[num_platforms_used],
        "purpose":[purpose],
        "avg_session_minutes":[avg_session_minutes],
        "night_usage":[night_usage],
        "addiction_level":[addiction_level],
        "screen_time_before_sleep":[screen_time_before_sleep]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(data)[0]

    st.success(f"Predicted Mental Health Score: {prediction:.2f}")
