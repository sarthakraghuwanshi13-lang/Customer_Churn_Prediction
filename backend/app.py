"""
Customer Churn Prediction - Flask API Server
Serves the trained ML model via REST endpoints.
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load Model & Encoders ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]
test_accuracy = model_data.get("test_accuracy", 0.78)
cv_accuracy = model_data.get("cv_accuracy", 0.84)

print(f"✅ Model loaded. Features: {feature_names}")
print(f"   Test Accuracy: {test_accuracy:.4f}")


# ─── Feature Schema (for frontend dropdowns) ────────────────────────────────
FEATURE_SCHEMA = {
    "gender": {"type": "categorical", "options": ["Female", "Male"], "label": "Gender"},
    "SeniorCitizen": {"type": "numerical", "options": [0, 1], "label": "Senior Citizen"},
    "Partner": {"type": "categorical", "options": ["No", "Yes"], "label": "Partner"},
    "Dependents": {"type": "categorical", "options": ["No", "Yes"], "label": "Dependents"},
    "tenure": {"type": "numerical", "min": 0, "max": 72, "label": "Tenure (months)"},
    "PhoneService": {"type": "categorical", "options": ["No", "Yes"], "label": "Phone Service"},
    "MultipleLines": {"type": "categorical", "options": ["No", "No phone service", "Yes"], "label": "Multiple Lines"},
    "InternetService": {"type": "categorical", "options": ["DSL", "Fiber optic", "No"], "label": "Internet Service"},
    "OnlineSecurity": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Online Security"},
    "OnlineBackup": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Online Backup"},
    "DeviceProtection": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Device Protection"},
    "TechSupport": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Tech Support"},
    "StreamingTV": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Streaming TV"},
    "StreamingMovies": {"type": "categorical", "options": ["No", "No internet service", "Yes"], "label": "Streaming Movies"},
    "Contract": {"type": "categorical", "options": ["Month-to-month", "One year", "Two year"], "label": "Contract"},
    "PaperlessBilling": {"type": "categorical", "options": ["No", "Yes"], "label": "Paperless Billing"},
    "PaymentMethod": {"type": "categorical", "options": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], "label": "Payment Method"},
    "MonthlyCharges": {"type": "numerical", "min": 18.25, "max": 118.75, "label": "Monthly Charges ($)"},
    "TotalCharges": {"type": "numerical", "min": 0, "max": 8684.80, "label": "Total Charges ($)"},
}


@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "endpoints": {
            "POST /predict": "Make a churn prediction",
            "GET /features": "Get feature schema for the form",
            "GET /model-info": "Get model performance metrics",
        }
    })


@app.route("/features", methods=["GET"])
def get_features():
    """Return the feature schema so the frontend can build the form dynamically."""
    return jsonify({
        "features": FEATURE_SCHEMA,
        "feature_order": feature_names,
    })


@app.route("/model-info", methods=["GET"])
def get_model_info():
    """Return model performance metrics."""
    return jsonify({
        "model_name": "Random Forest Classifier",
        "test_accuracy": round(test_accuracy, 4),
        "cv_accuracy": round(cv_accuracy, 4),
        "n_features": len(feature_names),
        "features": feature_names,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Accept customer data and return churn prediction."""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Build DataFrame from input
        input_data_df = pd.DataFrame([data])

        # Ensure all features are present
        missing = [f for f in feature_names if f not in input_data_df.columns]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Convert numeric fields
        for col in ["tenure", "SeniorCitizen"]:
            input_data_df[col] = input_data_df[col].astype(int)
        for col in ["MonthlyCharges", "TotalCharges"]:
            input_data_df[col] = input_data_df[col].astype(float)

        # Encode categorical features using the saved encoders
        for column, encoder in encoders.items():
            if column in input_data_df.columns:
                try:
                    input_data_df[column] = encoder.transform(input_data_df[column])
                except ValueError as e:
                    return jsonify({
                        "error": f"Invalid value for '{column}': {data.get(column)}. "
                                 f"Valid options: {list(encoder.classes_)}"
                    }), 400

        # Reorder columns to match training order
        input_data_df = input_data_df[feature_names]

        # Make prediction
        prediction = loaded_model.predict(input_data_df)
        pred_prob = loaded_model.predict_proba(input_data_df)

        result = {
            "prediction": int(prediction[0]),
            "churn": bool(prediction[0] == 1),
            "label": "Churn" if prediction[0] == 1 else "No Churn",
            "probability": {
                "no_churn": round(float(pred_prob[0][0]), 4),
                "churn": round(float(pred_prob[0][1]), 4),
            },
            "confidence": round(float(max(pred_prob[0])) * 100, 2),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
