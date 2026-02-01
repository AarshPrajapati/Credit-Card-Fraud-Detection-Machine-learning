from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from decision_engine import make_decision


# Initialize API
app = FastAPI(title="Fraud Detection API")

# Load saved model and scaler
model = joblib.load("./models/fraud_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Define a simple health check route
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

# Prediction route
@app.post("/predict")
def predict(transaction: dict):
    """
    transaction: dictionary of features for ONE transaction
    Example:
    {
      "V1": 0.123,
      "V2": -0.345,
      ...
      "Amount": 100.50
    }
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([transaction])
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Predict probability
    prob = model.predict_proba(X_scaled)[0][1]
    
    # Apply threshold
    threshold = 0.3
    decision_data = make_decision(prob)

    return {
        "fraud_probability": round(float(prob), 4),
        "risk_level": decision_data["risk_level"],
        "decision": decision_data["decision"]
    }


import shap

explainer = shap.TreeExplainer(model)

@app.post("/predict_explain")
def predict_explain(transaction: dict):
    df = pd.DataFrame([transaction])
    X_scaled = scaler.transform(df)
    
    prob = model.predict_proba(X_scaled)[0][1]
    decision = "FRAUD" if prob >= 0.3 else "APPROVE"
    
    # SHAP explanation
    shap_values = explainer.shap_values(df)
    shap_dict = {f"V{i+1}": float(shap_values[0][i]) for i in range(df.shape[1])}
    
    return {
        "fraud_probability": float(prob),
        "decision": decision,
        "shap_contributions": shap_dict
    }
