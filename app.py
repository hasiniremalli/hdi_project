"""
app.py
------
Flask backend for the HDI Prediction web application.

Routes:
    GET  /                 -> renders the frontend (templates/index.html)
    POST /predict           -> JSON API. Accepts the 4 raw indicators and
                                returns predicted HDI category + confidence
                                scores + the computed continuous HDI score.
    GET  /api/sample/<tier> -> returns example input values for a given tier
                                (used by the "Try an example" buttons)
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "hdi_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

FEATURES = [
    "life_expectancy",
    "mean_years_schooling",
    "expected_years_schooling",
    "gni_per_capita",
]

# Same UNDP-style normalization used to build the training data, so we can
# also show the user the underlying continuous HDI score alongside the
# model's categorical prediction.
LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15
EYS_MAX = 18
GNI_MIN, GNI_MAX = 100, 75000


def compute_hdi_score(life_expectancy, mean_schooling, expected_schooling, gni):
    lei = np.clip((life_expectancy - LE_MIN) / (LE_MAX - LE_MIN), 0, 1)
    ei = np.clip(
        ((mean_schooling / MYS_MAX) + (expected_schooling / EYS_MAX)) / 2, 0, 1
    )
    ii = np.clip(
        (np.log(max(gni, 1)) - np.log(GNI_MIN)) / (np.log(GNI_MAX) - np.log(GNI_MIN)),
        0,
        1,
    )
    return float((lei * ei * ii) ** (1 / 3))


SAMPLE_INPUTS = {
    "very_high": {
        "life_expectancy": 82.0,
        "mean_years_schooling": 12.5,
        "expected_years_schooling": 17.0,
        "gni_per_capita": 55000,
    },
    "high": {
        "life_expectancy": 74.0,
        "mean_years_schooling": 9.0,
        "expected_years_schooling": 13.5,
        "gni_per_capita": 15000,
    },
    "medium": {
        "life_expectancy": 66.0,
        "mean_years_schooling": 6.5,
        "expected_years_schooling": 10.5,
        "gni_per_capita": 5500,
    },
    "low": {
        "life_expectancy": 55.0,
        "mean_years_schooling": 3.0,
        "expected_years_schooling": 6.5,
        "gni_per_capita": 1400,
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        values = []
        for feat in FEATURES:
            if feat not in payload:
                return jsonify({"error": f"Missing field: {feat}"}), 400
            values.append(float(payload[feat]))

        life_expectancy, mean_schooling, expected_schooling, gni = values

        # Basic sanity-range validation
        if not (0 < life_expectancy <= 100):
            return jsonify({"error": "life_expectancy must be between 0 and 100"}), 400
        if not (0 <= mean_schooling <= 25):
            return jsonify({"error": "mean_years_schooling must be between 0 and 25"}), 400
        if not (0 <= expected_schooling <= 25):
            return jsonify({"error": "expected_years_schooling must be between 0 and 25"}), 400
        if not (0 < gni <= 200000):
            return jsonify({"error": "gni_per_capita must be between 0 and 200000"}), 400

        X = np.array([values])
        X_scaled = scaler.transform(X)

        pred_encoded = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        probabilities = model.predict_proba(X_scaled)[0]

        prob_dict = {
            label_encoder.classes_[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        hdi_score = compute_hdi_score(*values)

        return jsonify({
            "prediction": pred_label,
            "confidence": round(float(max(probabilities)), 4),
            "probabilities": prob_dict,
            "computed_hdi_score": round(hdi_score, 4),
            "input": dict(zip(FEATURES, values)),
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/sample/<tier>")
def sample(tier):
    tier = tier.lower().replace(" ", "_")
    if tier not in SAMPLE_INPUTS:
        return jsonify({"error": "Unknown tier. Use one of: very_high, high, medium, low"}), 404
    return jsonify(SAMPLE_INPUTS[tier])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
