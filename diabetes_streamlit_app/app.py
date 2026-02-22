import streamlit as st
import pickle
import numpy as np



import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))



st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 AI-Based Diabetes Risk Prediction")
st.write("Enter patient details below:")

# Inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level")
bloodPressure = st.number_input("Blood Pressure")
skinThickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    features = np.array([[pregnancies, glucose, bloodPressure,
                          skinThickness, insulin, bmi, dpf, age]])

    features = scaler.transform(features)

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({probability*100:.2f}%)")

    st.progress(float(probability))
