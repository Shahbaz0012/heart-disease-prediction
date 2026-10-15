"""Score one patient record with the saved model."""
import argparse
import json
import joblib
import pandas as pd

def predict(patient_json, model_path="models/heart_disease_model.joblib"):
    patient = pd.DataFrame([json.loads(patient_json)])
    probability = float(joblib.load(model_path).predict_proba(patient)[0, 1])
    return {"predicted_class": int(probability >= .5), "heart_disease_probability": round(probability, 4), "notice": "Educational model output only. It does not diagnose heart disease."}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--model", default="models/heart_disease_model.joblib")
    args = parser.parse_args()
    print(json.dumps(predict(args.patient, args.model), indent=2))
