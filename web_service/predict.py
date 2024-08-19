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

# Load environment variables from the .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Configuration: Define key variables from environment variables
TRACKING_SERVER_HOST = os.getenv('TRACKING_SERVER_HOST') # MLflow tracking server host
MODEL_NAME = os.getenv('MODEL_NAME') # Name of the model registered in MLflow
MODEL_CACHE_DIR = "/home/ubuntu/model_cache" # Directory to cache downloaded models locally

# Output the current configuration for debugging purposes
print(f"Tracking server: {TRACKING_SERVER_HOST}")
print(f"Model name: {MODEL_NAME}")
print(f"Model cache directory: {MODEL_CACHE_DIR}")

# Ensure the cache directory exists; create it if it doesn't
def ensure_cache_dir_exists():
    """
    Ensure the cache directory exists. If not, create it.
    This is useful for storing downloaded models locally to avoid repeated downloads.
    """
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)

# Construct the full path for the cached model based on its name and version
def get_cached_model_path(model_name, version):
    """
    Generate the file path for a cached model.

    Args:
        model_name (str): The name of the model.
        version (str): The version or alias of the model.

    Returns:
        str: The file path to the cached model.
    """
    return os.path.join(MODEL_CACHE_DIR, f"{model_name}_{version}.pkl")

# Save the model to the cache directory to avoid downloading it repeatedly
def save_model_to_cache(model, model_name, version):
    """
    Save the model to the local cache directory.

    Args:
        model: The model object to be saved.
        model_name (str): The name of the model.
        version (str): The version or alias of the model.

    Returns:
        None
    """
    ensure_cache_dir_exists()
    cache_path = get_cached_model_path(model_name, version)
    joblib.dump(model, cache_path)
    logging.info(f"Model saved to cache: {cache_path}")

# Load the model from the cache directory if it exists
def load_model_from_cache(model_name, version):
    """
    Load a model from the cache if available.

    Args:
        model_name (str): The name of the model.
        version (str): The version or alias of the model.

    Returns:
        model: The loaded model object, or None if not found.
    """
    cache_path = get_cached_model_path(model_name, version)
    if os.path.exists(cache_path):
        logging.info(f"Loading model from cache: {cache_path}")
        return joblib.load(cache_path)
    return None

# Fetch the model from the MLflow server, with caching to improve performance
def fetch_model(model_name=MODEL_NAME, alias="Production"):  
    """
    Fetch the model from MLflow, with a fallback to the cached model.

    Args:
        model_name (str): The name of the model to fetch.
        alias (str): The alias or version of the model to fetch (default: 'Production').

    Returns:
        model: The loaded model object.
    """
    # Attempt to load the model from the cache first
    cached_model = load_model_from_cache(model_name, alias)
    if cached_model is not None:
        logging.info(f"Model loaded from cache: {get_cached_model_path(model_name, alias)}")
        return cached_model

    # Set the MLflow tracking URI to connect to the server
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    model_artifact = f"models:/{model_name}@{alias}"

    try:
        # Load the model from MLflow using the model artifact path
        model = mlflow.sklearn.load_model(model_artifact)
        logging.info(f"The {model_name} model from {alias} is loaded from MLflow...")
        
        # Save the model to the cache for future use
        save_model_to_cache(model, model_name, alias)
        logging.info(f"Model saved to cache at: {get_cached_model_path(model_name, alias)}")
        
        return model

    except Exception as e:
        logging.error(f"Failed to load model from MLflow: {str(e)}")
        raise

# Function to make predictions based on input data
def predict(data):
    """
    Process uploaded data and generate predictions using the loaded model.

    Args:
        data (str): The CSV data as a string.

    Returns:
        tuple: A tuple containing the prediction results, the S3 URL where predictions are stored, and the HTTP status code.
    """
    if not data:
        return {"error": "The uploaded file is empty or could not be read."}, 400

    try:
        # Load the input CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(data))

        # Check if the DataFrame is empty and return an error if so
        if df.empty:
            return {"error": "The uploaded file is empty or contains no data."}, 400

        # Prepare the input data for prediction (e.g., feature extraction)
        feature_dicts, patient_ids = prepare_dictionaries(df)
        pipeline = fetch_model(model_name=MODEL_NAME, alias="Production")

        results = []
        for feature_dict, patient_id in zip(feature_dicts, patient_ids):
            # Transform the input features as needed by the pipeline's steps
            if 'dictvectorizer' in pipeline.named_steps:
                vectorized_features = pipeline.named_steps['dictvectorizer'].transform([feature_dict])
            else:
                vectorized_features = [feature_dict]

            if 'standardscaler' in pipeline.named_steps:
                scaled_features = pipeline.named_steps['standardscaler'].transform(vectorized_features)
            else:
                scaled_features = vectorized_features

            # Identify the classifier or model step in the pipeline
            classifier_step = next(
                (step_name for step_name in pipeline.named_steps 
                 if 'classifier' in step_name or 'model' in step_name), None
            )
            if classifier_step is None:
                raise ValueError("No classifier step found in the model pipeline.")

            # Make a prediction using the classifier
            pred = pipeline.named_steps[classifier_step].predict(scaled_features)
            admission = "admit" if pred == 1 else "discharge"

            results.append({"admission": admission, "patient_id": patient_id})
        
        # Save the predictions to S3 after they are generated
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

# Define the prediction endpoint that accepts a file upload and returns predictions
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint to handle POST requests for prediction.
    Expects a file upload with CSV data.

    Returns:
        JSON: A JSON response with predictions or an error message.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files['file']
        file_content = file.read().decode('utf-8')

        if not file_content.strip():
            return jsonify({"error": "The uploaded file is empty or could not be read."}), 400

        # Get the results and S3 path from the prediction function
        results, s3_path, status_code = predict(file_content)

        # Return both the predictions and the S3 URL in the JSON response
        return jsonify({"predictions": results, "s3_url": s3_path}), status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run the Flask app with debugging enabled, accessible to all network interfaces
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
