import os
import mlflow
from flask import Flask, request, jsonify

TRACKING_SERVER_HOST = os.getenv('TRACKING_SERVER_HOST', 'ec2-13-250-120-81.ap-southeast-1.compute.amazonaws.com')
MODEL_NAME = os.getenv('MODEL_NAME', 'xgboost_admission_prediction_model')

def fetch_model(model_name = MODEL_NAME, alias = "Production"):  

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(model_name)

    model_artifact = f"models:/{model_name}@{alias}"
    pipeline = mlflow.sklearn.load_model(model_artifact)
    print(f"The {model_name} model from {alias} is loaded...")

    return pipeline

def checked_features(data):
    numerical_features = [
        "age", "albumin_last", "albumin_max", "albumin_median", "albumin_min",
        "bloodculture,routine_count", "bloodculture,routine_last", "bloodculture,routine_npos",
        "cc_abdominalcramping", "cc_abdominaldistention", "cc_abdominalpain",
        "cc_abdominalpainpregnant", "cc_allergicreaction", "cc_bleeding/bruising",
        "cc_breastpain", "cc_chestpain", "cc_confusion", "cc_diarrhea", "cc_dizziness",
        "cc_fall>65", "cc_fever", "cc_hallucinations", "cc_headache", "cc_hypertension",
        "cc_hypotension", "cc_irregularheartbeat", "cc_nausea", "cc_overdose-accidental",
        "cc_overdose-intentional", "cc_poisoning", "cc_rapidheartrate", "cc_rectalbleeding",
        "cc_strokealert", "cc_unresponsive", "cc_urinaryretention", "cktotal_last", 
        "cktotal_max", "cktotal_median", "cktotal_min", "d-dimer_last", "d-dimer_max", 
        "d-dimer_median", "d-dimer_min", "esi", "n_admissions", "n_edvisits", "n_surgeries", 
        "platelets_last", "platelets_max", "platelets_median", "platelets_min", 
        "rbc_last", "rbc_max", "rbc_median", "rbc_min", "triage_vital_dbp", 
        "triage_vital_hr", "triage_vital_o2", "triage_vital_o2_device", "triage_vital_rr", 
        "triage_vital_sbp", "triage_vital_temp", "troponini(poc)_last", "troponini(poc)_max", 
        "troponini(poc)_median", "troponini(poc)_min", "troponint_last", "troponint_max", 
        "troponint_median", "troponint_min", "urineculture,routine_count", 
        "urineculture,routine_last", "urineculture,routine_npos", "viralinfect", 
        "wbc_last", "wbc_max", "wbc_median", "wbc_min"
    ]

    categorical_features = ['arrivalmode', 'gender', 'previousdispo']

    features = {}

    # Handle numerical features
    for feature in numerical_features:
        features[feature] = data.get(feature, 0)

    # Handle categorical features
    for feature in categorical_features:
        value = data.get(feature, "")
        features[feature] = str(value).lower().replace(' ', '_') if value else ""

    return features

def predict(data):
    features = checked_features(data)
    pipeline = fetch_model(model_name = MODEL_NAME, alias = "Production")
    vectorized_features = pipeline.named_steps['dictvectorizer'].transform([features])
    scaled_features = pipeline.named_steps['standardscaler'].transform(vectorized_features)
    pred = pipeline.named_steps['xgbclassifier'].predict(scaled_features)
    admission = "admit" if pred == 1 else "discharge"
    return admission

app = Flask('admission-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    pred = predict(data)    

    result = {
        "admission": pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)