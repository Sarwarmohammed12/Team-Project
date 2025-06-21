import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction App")

# Get user input
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Gender", ["Male", "Female"])
cholesterol = st.slider("Cholesterol", 100, 400, 200)
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

# Convert gender to numeric
sex_value = 1 if sex == "Male" else 0

# Create input DataFrame
data = pd.DataFrame([[
    age, cholesterol, resting_bp, max_hr, oldpeak, sex_value
]], columns=["Age", "Cholesterol", "RestingBP", "MaxHR", "Oldpeak", "Sex"])

# Predict
if st.button("Predict"):
    result = model.predict(data)[0]
    st.success("Heart Disease Detected" if result == 1 else "No Heart Disease")
