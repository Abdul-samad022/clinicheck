from flask import Flask, request, jsonify, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)
#model = joblib.load("rf_diagnosis_model.joblib")
BASE_DIR = os.path.dirname(__file__)  # directory of app.py
model_path = os.path.join(BASE_DIR, 'rf_diagnosis_model.joblib')
model = joblib.load(model_path)

SYMPTOMS = [
    "symptom_fever", "symptom_cough", "symptom_fatigue", "symptom_headache",
    "symptom_nausea", "symptom_vomiting", "symptom_diarrhea",
    "symptom_sore_throat", "symptom_shortness_of_breath"
]
FEATURE_ORDER = ["age", "temperature", "heart_rate"] + SYMPTOMS + ["sex", "comorb_diabetes", "comorb_htn"]

@app.route("/")
def home():
    return render_template("index.html", symptoms=SYMPTOMS)

@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type == "application/json":
        data = request.get_json()
    else:
        # From HTML form
        data = {f: request.form.get(f) for f in FEATURE_ORDER}
        # Convert numeric fields
        for f in ["age", "temperature", "heart_rate", "comorb_diabetes", "comorb_htn"] + SYMPTOMS:
            data[f] = int(data.get(f, 0))
        data["sex"] = data["sex"]  # M or F

    row = {}
    for f in FEATURE_ORDER:
        val = data.get(f, 0 if f.startswith("symptom_") else None)
        if val is None:
            return jsonify({"error": f"Missing field: {f}"}), 400
        row[f] = val

    row["sex"] = 0 if row["sex"] == "M" else 1
    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    probs = model.predict_proba(X)[0]
    labels = model.classes_
    results = sorted(zip(labels, probs), key=lambda x: -x[1])

    if request.content_type == "application/json":
        return jsonify({
            "predictions": [
                {"diagnosis": lab, "probability": round(float(prob), 4)}
                for lab, prob in results
            ],
            "disclaimer": "This is a synthetic demo. Not for real medical use."
        })
    else:
        return render_template("result.html", results=results)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)