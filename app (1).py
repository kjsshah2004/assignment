import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📞 Telecom Customer Churn Predictor")
st.write("Enter customer details below to predict their likelihood of churning.")

# Load the saved model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}. Please ensure model.joblib and scaler.joblib are in the same folder.")
    st.stop()

# Input components
tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=50.0)
SeniorCitizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
senior_val = 1 if SeniorCitizen == "Yes" else 0

gender = st.selectbox("Gender", ["Female", "Male"])
gender_Male = 1 if gender == "Male" else 0

partner = st.selectbox("Has Partner?", ["No", "Yes"])
Partner_Yes = 1 if partner == "Yes" else 0

dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
Dependents_Yes = 1 if dependents == "Yes" else 0

# Add placeholder inputs for other encoded features to match the expected shape (29 features)
inputs = {
    "SeniorCitizen": senior_val,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "gender_Male": gender_Male,
    "Partner_Yes": Partner_Yes,
    "Dependents_Yes": Dependents_Yes
}

# Fill in remains of the 29 features with default values (0)
expected_features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 
    'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

features_dict = {}
for feat in expected_features:
    features_dict[feat] = inputs.get(feat, 0)

# Build the dataframe
input_df = pd.DataFrame([features_dict])

if st.button("Predict Churn"):
    # Process data
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    
    st.subheader("Prediction Results:")
    if prediction == 1:
        st.error(f"This customer is likely to CHURN (Probability: {probability:.2%})")
    else:
        st.success(f"This customer is likely to STAY (Probability of churn: {probability:.2%})")
