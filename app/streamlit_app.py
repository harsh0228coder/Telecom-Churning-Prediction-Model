import streamlit as st
import pandas as pd
import pickle


with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\gender_le.pkl', 'rb') as f:
    gender_le = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\contract_le.pkl', 'rb') as f:
    contract_le = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\paymentmethod_le.pkl', 'rb') as f:
    paymentmethod_le = pickle.load(f)

st.title("Telecom Customer Churn Prediction")
gender = st.selectbox("Gender", ['Male', 'Female'])
contract = st.selectbox("Contract", ['Month-to-month', 'Two year', 'One year'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
tenure = st.number_input("Tenure (in months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn"):
    gender_encoded = gender_le.transform([gender])
    contract_encoded = contract_le.transform([contract])
    payment_method_encoded = paymentmethod_le.transform([payment_method])

    paperless_billing_encoded = 1 if paperless_billing == 'Yes' else 0
    prediction = model.predict([[tenure, paperless_billing_encoded, monthly_charges, total_charges, gender_encoded[0], contract_encoded[0], payment_method_encoded[0]]])
    if prediction[0] == 1:
        st.write("Customer is going to churn")
    else:
        st.write("Customer is not going to churn")