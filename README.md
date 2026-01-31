# 🕵️ Credit Card Fraud Detection System

## 📌 Overview
This project is an **end-to-end machine learning system** for detecting fraudulent credit card transactions using supervised learning on highly imbalanced financial data.

The goal is to demonstrate **real-world ML engineering skills**, including:
- Data understanding and exploration
- Model training and evaluation
- Handling class imbalance
- Model explainability
- Serving predictions via an API
- Deployment-ready project structure

---

## 🧠 Problem Statement
Credit card fraud is a **rare but high-impact** problem.

Key challenges:
- Fraud cases represent a very small percentage of total transactions
- Accuracy alone is misleading due to class imbalance
- Models must balance **fraud recall** and **false positives**

This project focuses on building a model that **effectively identifies fraudulent transactions while remaining interpretable and deployable**.

---

## 📊 Dataset

📌 The dataset is **not included** in this repository due to size limitations.  

- **Source:** [European Credit Card Transactions Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)  
- **Total Transactions:** 284,807  
- **Fraud Rate:** ~0.17%  
- **Target Column:** `Class` (1 = Fraud, 0 = Normal)  

### Features

- `V1` – `V28`: PCA-transformed and anonymized features (for privacy)  
- `Amount`: Transaction amount  

---

## 📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── api/
│   └── main.py           # FastAPI inference service
│
├── models/
│   ├── fraud_model.pkl   # Trained ML model
│   └── scaler.pkl        # Feature scaler
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── data/
│   └── creditcard.csv    # Dataset not included
│
├── requirements.txt
├── Dockerfile
└── README.md


---

## 📈 Data Exploration
Exploratory data analysis is performed in:


This notebook covers:
- Dataset overview and schema
- Class imbalance analysis
- Transaction amount distribution
- Initial feature understanding

---

## ⚙️ Machine Learning Approach
- Model used: **XGBoost Classifier**
- Feature scaling: StandardScaler
- Class imbalance handled using `scale_pos_weight`
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Precision-Recall AUC

The model is trained to **prioritize fraud recall** while keeping false positives under control.

---

### 📊 Model Performance & Threshold Tuning

Due to severe class imbalance, multiple probability thresholds were evaluated.

| Threshold | Fraud Precision | Fraud Recall | Fraud F1 |
|---------|----------------|--------------|----------|
| 0.5 | 0.88 | 0.84 | 0.86 |
| 0.4 | 0.85 | 0.84 | 0.85 |
| 0.3 | 0.81 | 0.85 | 0.83 |
| 0.2 | 0.77 | 0.86 | 0.81 |

Lower thresholds improve fraud recall at the cost of increased false positives.
Threshold selection can be adjusted based on business risk tolerance.


---

## 🔍 Model Explainability
SHAP (SHapley Additive exPlanations) is used to:
- Understand feature contributions
- Explain individual predictions
- Improve trust and transparency

This is critical for financial and regulated domains.

---

## 🚀 API Service
A **FastAPI-based REST API** is provided for real-time fraud prediction.

### Example Endpoint

### Sample Request
```json
{
  "V1": -1.23,
  "V2": 0.45,
  "...": "...",
  "V28": 0.12,
  "Amount": 250.0
}

### Sample Response
{
  "fraud_probability": 0.42,
  "decision": "REVIEW"
}
