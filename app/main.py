from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
 
 
app = Flask(__name__)
 
# Load the trained model
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\gender_le.pkl', 'rb') as f:
    gender_le = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\contract_le.pkl', 'rb') as f:
    contract_le = pickle.load(f)
with open(r'C:\Users\91727\OneDrive\Desktop\Data Science Project\Model\version1\paymentmethod_le.pkl', 'rb') as f:
    paymentmethod_le = pickle.load(f)
 
@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
    """
    gender : Male/ Female
    contract: Month-to-month/ Two year/ One year
    payment-method: Electronic check/ Mailed check/ Bank transfer (automatic)/ Credit card (automatic)
    paperless_billing: Yes/No
    tenure: no. of months the customer has stayed with the company
    monthly_charges: The amount charged to the customer monthly
    total_charges: The total amount charged to the customer
    --------------------------------------------------------------------------
    Sample Input:
    {
        "gender": "Female",
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "paperless_billing": "Yes",
        "monthly_charges": 29.85,
        "total_charges": 60.85,
        "tenure": 3,
        }
    """
    data = request.get_json(force=True)
    gender = gender_le.transform([data["gender"]])
    contract = contract_le.transform([data["contract"]])
    payment_method = paymentmethod_le.transform([data["payment_method"]])
 
    paperless_billing = 1 if data["paperless_billing"] == 'Yes' else 0
    tenure = data["tenure"]
    monthly_charges = data["monthly_charges"]
    total_charges = data["total_charges"]
 
    print(f"gender {gender}, contract {contract} paymentmethod {payment_method}")
 
    prediction = model.predict([[tenure,paperless_billing,monthly_charges,
                                total_charges,gender[0],contract[0],payment_method[0]]])
    if prediction[0] == 1:
        return "Customer is going to churn"
    else:
        return "Customer is not going to churn"



if __name__ == '__main__':
    app.run(debug=True)