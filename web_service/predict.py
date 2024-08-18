import mlflow
from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import io
from utils import prepare_dictionaries, save_predictions_to_s3
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Configuration
TRACKING_SERVER_HOST = os.getenv('TRACKING_SERVER_HOST')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_CACHE_DIR = "/home/ubuntu/model_cache"  # Directory to cache models

print(f"Tracking server: {TRACKING_SERVER_HOST}")
print(f"Model name: {MODEL_NAME}")
print(f"Model cache directory: {MODEL_CACHE_DIR}")

# Ensure cache directory exists
def ensure_cache_dir_exists():
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)

def get_cached_model_path(model_name, version):
    return os.path.join(MODEL_CACHE_DIR, f"{model_name}_{version}.pkl")

def save_model_to_cache(model, model_name, version):
    ensure_cache_dir_exists()
    cache_path = get_cached_model_path(model_name, version)
    joblib.dump(model, cache_path)
    logging.info(f"Model saved to cache: {cache_path}")

def load_model_from_cache(model_name, version):
    cache_path = get_cached_model_path(model_name, version)
    if os.path.exists(cache_path):
        logging.info(f"Loading model from cache: {cache_path}")
        return joblib.load(cache_path)
    return None

# Fetch model from MLflow
def fetch_model(model_name=MODEL_NAME, alias="Production"):  
    cached_model = load_model_from_cache(model_name, alias)
    if cached_model is not None:
        logging.info(f"Model loaded from cache: {get_cached_model_path(model_name, alias)}")
        return cached_model

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    model_artifact = f"models:/{model_name}@{alias}"

    try:
        # Load the model from MLflow
        model = mlflow.sklearn.load_model(model_artifact)
        logging.info(f"The {model_name} model from {alias} is loaded from MLflow...")
        
        # Save to cache
        save_model_to_cache(model, model_name, alias)
        logging.info(f"Model saved to cache at: {get_cached_model_path(model_name, alias)}")
        
        return model

    except Exception as e:
        logging.error(f"Failed to load model from MLflow: {str(e)}")
        raise

# Prediction function
def predict(data):
    if not data:
        return {"error": "The uploaded file is empty or could not be read."}, 400

    try:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(data))

        # Check if the DataFrame is empty
        if df.empty:
            return {"error": "The uploaded file is empty or contains no data."}, 400

        # Prepare the data for prediction
        feature_dicts, patient_ids = prepare_dictionaries(df)
        pipeline = fetch_model(model_name=MODEL_NAME, alias="Production")

        results = []
        for feature_dict, patient_id in zip(feature_dicts, patient_ids):
            # Process features through pipeline
            if 'dictvectorizer' in pipeline.named_steps:
                vectorized_features = pipeline.named_steps['dictvectorizer'].transform([feature_dict])
            else:
                vectorized_features = [feature_dict]

            if 'standardscaler' in pipeline.named_steps:
                scaled_features = pipeline.named_steps['standardscaler'].transform(vectorized_features)
            else:
                scaled_features = vectorized_features

            # Identify classifier step
            classifier_step = next(
                (step_name for step_name in pipeline.named_steps 
                 if 'classifier' in step_name or 'model' in step_name), None
            )
            if classifier_step is None:
                raise ValueError("No classifier step found in the model pipeline.")

            # Make prediction
            pred = pipeline.named_steps[classifier_step].predict(scaled_features)
            admission = "admit" if pred == 1 else "discharge"

            results.append({"admission": admission, "patient_id": patient_id})
        
        # Call the S3 upload function after generating predictions
        s3_url = save_predictions_to_s3(results)

        return results, s3_url, 200

    except pd.errors.EmptyDataError:
        return {"error": "The uploaded file is empty or not valid CSV."}, 400
    except pd.errors.ParserError:
        return {"error": "The uploaded file cannot be parsed as CSV."}, 400
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500

# Flask app setup
app = Flask('admission-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files['file']
        file_content = file.read().decode('utf-8')

        if not file_content.strip():
            return jsonify({"error": "The uploaded file is empty or could not be read."}), 400

        # Get the results and S3 path
        results, s3_path, status_code = predict(file_content)

        # Return both results and s3_path in the JSON response
        return jsonify({"predictions": results, "s3_url": s3_path}), status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
