import streamlit as st
import pandas as pd
import numpy as np
import joblib
#LOAD SERIALIZED PIPELINE ARTIFACTS
@st.cache_resource
def load_assets():
    model = joblib.load("best_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler
try:
    model, scaler = load_assets()
    st.title("Healthcare Disease Risk predictor")
    st.write("Input patient metrics below to evaluate clinical outcomes instantly.")
    st.sidebar.header("Patient Demographics & Vitals")
    age = st.sidebar.slider("Age",1, 100, 45)
    bmi = st.sidebar.number_input("BMI Index (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5)
    blood_pressure = st.sidebar.slider("Systollic Blood pressure (mmHg)", 80, 200, 120)
    cholesterol = st.sidebar.slider("Cholesterol Level (mg/dL)", 100, 400, 190)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 300, 95)
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    smoking = st.sidebar.selectbox("Smoking", options=["Never Smoked", "Former Smoker", "Current Smoker"])
    st.subheader("Prediction Analysis")
    if st.button("Predict Risk"):
      gender_encoded = 1 if gender == "Mlae" else 0
      smoking_map = {"Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2}
      smoking_encoded = smoking_map[smoking]
      st.sidebar.header("Lifestyle Factors")
      alcohol = st.sidebar.selectbox("Alcohol Consumption", options=["Never", "Occasional", "Frequent"])
      alcohol_map = {"Never": 0, "Occasional": 1, "Frequent": 2}
      alcohol_encoded = alcohol_map[alcohol]
      exercise = st.sidebar.selectbox("Exercise Frequently", options=["None","1-2 days/week", "3+ days/week"])
      exercise_map = {"None": 0, "1-2 days/week": 1, "3+ days/week": 2}
      exercise_encoded = exercise_map[exercise]
      st.sidebar.header("Medical & Family History")
      conditions = ["Family History", "Heart Disease", "Diabetes", "Stroke", "Kidney Disease", "Cancer", "Alzheimer's Disease", "COPD", "Liver Disease", "Parkinson's Disease", "Tuberculosis"]
      encoded_conditions = {}
      for condition in conditions:
           choice = st.sidebar.selectbox(f"History of {condition}?", options=["No","Yes"])
           encoded_conditions[condition] = 1
           if choice == "Yes":
                encoded_conditions[condition]
           else:
                encoded_conditions[condition]
      stroke_input = st.selectbox("Has this patient ever had a stroke?",["No","Yes"])
      input_df = pd.DataFrame([{'Age': age, 'BMI': bmi, 'Blood Pressure': blood_pressure, 'Cholesterol': cholesterol, 'Glucose': glucose, 'Gender': gender_encoded, 'Smoking': smoking_encoded, 'Alcohol Consumption': alcohol_encoded, 'Exercise': exercise_encoded, 'Family History':encoded_conditions["Family History"], 'Heart Disease': encoded_conditions["Heart Disease"], "Alzheimer's Disease":encoded_conditions["Alzheimer's Disease"], 'Cancer': encoded_conditions["Cancer"], 'COPD': encoded_conditions["COPD"], 'Diabetes': encoded_conditions["Diabetes"], 'Kidney Disease': encoded_conditions["Kidney Disease"], 'Liver Disease': encoded_conditions["Liver Disease"], "Parkinson's Disease": encoded_conditions["Parkinson's Disease"], 'Tuberculosis': encoded_conditions["Tuberculosis"], 'Stroke': stroke_input}])
      features_for_model = input_df.drop(columns=['Stroke'])
      prediction = model.predict(features_for_model)
      prediction_proba = model.predict_proba(features_for_model)[0][1]
      st.write("---")
      if prediction[0] == 1:
            st.error(f"High Risk Detected (Probability: {prediction_proba:.2%})")
      else:
            st.success(f"Low Risk Detected (Probability of High Risk: {prediction_proba:.2%})")
except Exception as e:
         st.error(f"Error loading assets or running prediction: {e}")
         
