"""
Customer Churn Prediction - Model Training Script
Replicates the Colab notebook pipeline and generates model + encoder pickle files.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# ─── 1. Load Data ───────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# ─── 2. Data Cleaning ───────────────────────────────────────────────────────
# Drop customerID column
df = df.drop(columns=["customerID"])

# Handle blank TotalCharges (space strings) → replace with 0 then convert to float
df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Encode target column
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})

print(f"After cleaning: {df.shape}")
print(f"Churn distribution:\n{df['Churn'].value_counts()}\n")

# ─── 3. Label Encoding ──────────────────────────────────────────────────────
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Categorical columns to encode: {categorical_columns}")

encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le
    print(f"  {column}: {list(le.classes_)}")

# Save encoders
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")
with open(ENCODERS_PATH, "wb") as f:
    pickle.dump(encoders, f)
print(f"\n✅ Encoders saved to {ENCODERS_PATH}")

# ─── 4. Train/Test Split ────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training class distribution:\n{y_train.value_counts()}")

# ─── 5. SMOTE Oversampling ──────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {X_train_smote.shape}")
print(f"SMOTE class distribution:\n{y_train_smote.value_counts()}")

# ─── 6. Model Comparison ────────────────────────────────────────────────────
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
}

print("\n" + "=" * 70)
print("MODEL COMPARISON (5-fold Cross Validation)")
print("=" * 70)

cv_scores = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = np.mean(scores)
    print(f"  CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

best_model_name = max(cv_scores, key=cv_scores.get)
print(f"\n🏆 Best Model: {best_model_name} (CV Accuracy: {cv_scores[best_model_name]:.4f})")

# ─── 7. Train Final Model (Random Forest) ───────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING FINAL MODEL: Random Forest")
print("=" * 70)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# Evaluate on test data
y_test_pred = rfc.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")

# ─── 8. Save Model ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "customer_churn_model.pkl")
model_data = {
    "model": rfc,
    "features_names": X.columns.tolist(),
    "test_accuracy": test_accuracy,
    "cv_accuracy": cv_scores["Random Forest"],
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"   Features: {X.columns.tolist()}")
print("\n🎉 Training complete!")
