# 🔮 ChurnSense AI — Customer Churn Prediction

A full-stack Machine Learning project that predicts customer churn for telecom companies using a **Random Forest** model, served via a **Flask API** with a premium dark-themed **web frontend**.

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco dataset (7,043 records)
├── backend/
│   ├── train_model.py          # ML training pipeline
│   ├── app.py                  # Flask REST API
│   ├── customer_churn_model.pkl # Trained model (generated)
│   ├── encoders.pkl            # Label encoders (generated)
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── index.html              # Main UI
│   ├── style.css               # Premium dark theme
│   └── script.js               # API interaction & animations
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This generates `customer_churn_model.pkl` and `encoders.pkl`.

### 3. Start the API Server
```bash
python app.py
```
Server runs at `http://127.0.0.1:5000`.

### 4. Open the Frontend
Open `frontend/index.html` in your browser. Fill in customer details and click **Predict Churn Risk**.

## 🤖 ML Pipeline

1. **Data Cleaning** — Drop `customerID`, handle blank `TotalCharges`
2. **Label Encoding** — Encode 15 categorical features
3. **SMOTE** — Balance class distribution (73% No Churn → 50/50)
4. **Model Comparison** — Decision Tree, Random Forest, XGBoost
5. **Final Model** — Random Forest (~84% CV accuracy, ~78% test accuracy)

## 📡 API Endpoints

| Method | Endpoint       | Description                    |
|--------|---------------|--------------------------------|
| GET    | `/`           | API info                       |
| GET    | `/features`   | Feature schema for form inputs |
| GET    | `/model-info` | Model accuracy & metadata      |
| POST   | `/predict`    | Churn prediction               |

### Example Prediction Request
```json
POST /predict
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

## 📊 Dataset

**Telco Customer Churn** — IBM Sample Dataset  
- 7,043 customers × 21 features  
- Binary target: Churn (Yes/No)  
- Features: demographics, services, billing, contract info
