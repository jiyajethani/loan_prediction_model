import joblib
import pandas as pd


# Load trained model
model = joblib.load("../model.pkl")


# Sample input (change values to test)
sample = pd.DataFrame({
    "Gender": ["Male"],
    "Married": ["Yes"],
    "Dependents": ["1"],
    "Education": ["Graduate"],
    "Self_Employed": ["No"],
    "Income": [5000],
    "LoanAmount": [120],
    "Credit_History": [1],
    "Property_Area": ["Urban"]
})


# Prediction
prediction = model.predict(sample)

result = "Approved" if prediction[0] == 1 else "Rejected"

print("\n💡 Loan Prediction Result:", result)
