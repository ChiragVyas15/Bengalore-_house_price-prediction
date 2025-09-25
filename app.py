import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and label encoder
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load df10 to get location columns
df10 = pd.read_csv("df10.csv")  # Save df10 to CSV after preprocessing in your notebook

# Get location columns
location_columns = [col for col in df10.columns if col.startswith("location_")]
feature_columns = ["total_sqft", "bath", "bhk"] + location_columns

st.title("Bangalore House Price Prediction")

# User input
total_sqft = st.number_input("Total Square Feet", min_value=100)
bath = st.number_input("Number of Bathrooms", min_value=1)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1)
location = st.selectbox("Location", options=[col.replace("location_", "") for col in location_columns])

# Prepare input for prediction
input_data = [total_sqft, bath, bhk]
for loc in location_columns:
    if loc == f"location_{location}":
        input_data.append(1)
    else:
        input_data.append(0)

if st.button("Predict Price"):
    prediction = model.predict([input_data])
    st.success(f"Estimated Price: ₹ {prediction[0]} Lakhs",icon="💰")