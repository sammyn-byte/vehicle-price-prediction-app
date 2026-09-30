
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('vehicle_price_model.pkl')
scaler = joblib.load('vehicle_price_scaler.pkl')

# Page configuration
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Title
st.title("🚗 Vehicle Selling Price Predictor")

st.write(
    "Enter the vehicle details below to estimate its selling price."
)

# User inputs
vehicle_age = st.number_input(
    "Vehicle Age (years)",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)

engine = st.number_input(
    "Engine Capacity (cc)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=50
)

max_power = st.number_input(
    "Maximum Power (bhp)",
    min_value=20.0,
    max_value=500.0,
    value=100.0,
    step=1.0
)

# Prediction
if st.button("Predict Selling Price"):

    # Arrange inputs in the same order as the training features
    input_data = pd.DataFrame({
        'max_power': [max_power],
        'engine': [engine],
        'vehicle_age': [vehicle_age]
    })

    # Standardize the input
    input_scaled = scaler.transform(input_data)

    # Predict log-transformed selling price
    log_prediction = model.predict(input_scaled)[0]

    # Convert prediction back to original price scale
    prediction = np.exp(log_prediction)

    # Display prediction
    st.success(
        f"Estimated Selling Price: {prediction:,.2f}"
    )
