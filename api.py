"""
Lung Cancer Prediction — Flask API
====================================
Loads trained models from saved_models/ and serves predictions
via a simple REST endpoint for the HTML frontend.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────────
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_models"
)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "gaussian_nb": "Gaussian Naive Bayes",
    "multinomial_nb": "Multinomial Naive Bayes",
    "svc": "Support Vector Classifier",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP (Neural Network)",
    "gradient_boosting": "Gradient Boosting",
}

# Feature names must EXACTLY match the training data columns
# (note: FATIGUE and ALLERGY have trailing spaces in the CSV)
FEATURE_NAMES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC DISEASE", "FATIGUE ", "ALLERGY ",
    "WHEEZING", "ALCOHOL CONSUMING", "COUGHING",
    "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN",
]

# ── Load Models ────────────────────────────────────────────────
models = {}
for key in MODEL_DISPLAY_NAMES:
    filepath = os.path.join(MODELS_DIR, f"{key}.pkl")
    if os.path.exists(filepath):
        models[key] = joblib.load(filepath)
        print(f"  [OK] Loaded {key}")
    else:
        print(f"  [MISSING] {filepath}")


# ── Routes ─────────────────────────────────────────────────────
@app.route("/models", methods=["GET"])
def list_models():
    """Return all available models."""
    available = {k: MODEL_DISPLAY_NAMES[k] for k in models}
    return jsonify(available)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with the 15 features + model key, return prediction.

    Expected JSON body:
    {
        "model": "random_forest",
        "gender": 1,          # 1=Male, 0=Female
        "age": 55,
        "smoking": 1,         # 0 or 1
        "yellow_fingers": 0,
        "anxiety": 1,
        "peer_pressure": 0,
        "chronic_disease": 1,
        "fatigue": 1,
        "allergy": 0,
        "wheezing": 1,
        "alcohol": 1,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 0,
        "chest_pain": 1
    }
    """
    data = request.get_json(force=True)

    # Validate model selection
    model_key = data.get("model", "random_forest")
    if model_key not in models:
        return jsonify({"error": f"Model '{model_key}' not found"}), 400

    # Build feature vector in training column order
    try:
        feature_values = [
            int(data.get("gender", 0)),
            int(data.get("age", 50)),
            int(data.get("smoking", 0)),
            int(data.get("yellow_fingers", 0)),
            int(data.get("anxiety", 0)),
            int(data.get("peer_pressure", 0)),
            int(data.get("chronic_disease", 0)),
            int(data.get("fatigue", 0)),
            int(data.get("allergy", 0)),
            int(data.get("wheezing", 0)),
            int(data.get("alcohol", 0)),
            int(data.get("coughing", 0)),
            int(data.get("shortness_of_breath", 0)),
            int(data.get("swallowing_difficulty", 0)),
            int(data.get("chest_pain", 0)),
        ]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

    model = models[model_key]
    prediction = int(model.predict(features_df)[0])

    return jsonify({
        "prediction": prediction,
        "risk": "HIGH" if prediction == 1 else "LOW",
        "model": MODEL_DISPLAY_NAMES.get(model_key, model_key),
    })


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Lung Cancer Prediction API")
    print(f"Loaded {len(models)}/{len(MODEL_DISPLAY_NAMES)} models")
    print(f"{'='*50}\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
