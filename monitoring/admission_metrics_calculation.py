import datetime
import time
import random
import logging 
import uuid
import pandas as pd
import os
import psycopg
import psycopg2
import mlflow
import numpy as np
import warnings
from config import numerical_features, categorical_features
from datetime import date

from prefect import task, flow
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric

from config import numerical_features, categorical_features, target
from data_processing import split_dataset, prepare_dictionaries, prepare_reference_and_raw_data
from initialize_mlflow import initialize 
from mlflow_training import compare_model_performances, compare_models_performance_and_select_best
from hyperparameter_tuning import hyperparameter_tuning
from mlflow_model_testing_registry import evaluate_models

# Suppress specific numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
DBNAME = os.getenv("DBNAME")
MODEL_NAME = os.getenv("MODEL_NAME")
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")
DATASET_PATH = os.getenv("LOCALPATH") # LOCALPATH for reading local csv file or NONE for downloading dataset from S3 bucket

SEND_TIMEOUT = 10

# SQL statement for creating a table to log model drift metrics
create_table_statement = """
drop table if exists admission_metrics;
create table admission_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""

def fetch_model(model_name=MODEL_NAME, alias="Production"): 
    """
    Fetch the latest version of a registered MLflow model.

    Args:
        model_name (str): The name of the model to fetch.
        alias (str): The stage of the model (e.g., "Production").

    Returns:
        pipeline (sklearn.pipeline.Pipeline): The loaded machine learning model pipeline.
    """ 
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    model_artifact = f"models:/{model_name}@{alias}"
    pipeline = mlflow.sklearn.load_model(model_artifact)
    print(f"The {model_name} model from {alias} is loaded...")
    return pipeline

# Initialize the model pipeline globally
pipeline = fetch_model()

def checked_features(data):
    """
    Prepare the input features by handling missing or empty values.

    Args:
        data (dict): The input data with raw features.

    Returns:
        dict: A dictionary with processed numerical and categorical features.
    """
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
    """
    Make a prediction using the preloaded model pipeline.

    Args:
        data (dict): The input data for prediction.

    Returns:
        str: The predicted outcome, either "admit" or "discharge".
    """
    features = checked_features(data)
    vectorized_features = pipeline.named_steps['dictvectorizer'].transform([features])
    scaled_features = pipeline.named_steps['standardscaler'].transform(vectorized_features)
    pred = pipeline.named_steps['xgbclassifier'].predict(scaled_features)
    admission = "admit" if pred == 1 else "discharge"
    return admission

begin = datetime.datetime.now()

def log_metrics_to_postgres(model_name, drift_score, num_drifted_columns, share_missing_values):
    """
    Log model drift metrics to a PostgreSQL database.

    Args:
        model_name (str): The name of the model.
        drift_score (float): The drift score of the model predictions.
        num_drifted_columns (int): The number of columns with detected drift.
        share_missing_values (float): The proportion of missing values in the data.
    """
    conn = psycopg2.connect(
        host=HOST,
        database=DBNAME,
        user=USER,
        password=PASSWORD
    )
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO model_metrics (timestamp, model_name, drift_score, num_drifted_columns, share_missing_values)
    VALUES (%s, %s, %s, %s, %s)
    """
    data = (datetime.datetime.now(), model_name, drift_score, num_drifted_columns, share_missing_values)
    cursor.execute(insert_query, data)

    conn.commit()
    cursor.close()
    conn.close()

def retraing(drift_score, num_drifted_columns, share_missing_values):
    """
    Retrain the model if drift exceeds a predefined threshold.

    Args:
        drift_score (float): The drift score triggering the retraining.
        num_drifted_columns (int): The number of drifted columns.
        share_missing_values (float): The proportion of missing values.

    Returns:
        str: The name of the newly trained model.
    """
    print("\n")
    print("THRESHOLD FOR PREDICTION EXCEEDED!!!\n")
    print("CONDITIONAL WORKFLOW INITIATED!!!\n")

    # Prepare dataset for retraining
    print("Training dataset is prepared...")
    train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, train_df, valid_df, test_df, path = split_dataset(
        numerical_features, categorical_features, target, path=DATASET_PATH
    )

    # Initialize mlflow
    print("MLflow is initialized...")
    initialize()

    # Compare model performances and choose the best model
    print("Model performances are compared...")
    best_model = compare_model_performances(train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path, numerical_features, categorical_features, target)

    # Hyperparameter tuning for the best model
    print("Best model is fine tuned...")
    model_name = f"admission_prediction_{best_model}_{date.today()}"
    hyperparameter_tuning(best_model, model_name, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path)

    # Evaluate models and register the best performing one
    print("Model is tested and registered...")
    evaluate_models(model_name, test_dicts, y_test)

    # Evaluate tuned model against the former production model
    new_model_name = compare_models_performance_and_select_best(MODEL_NAME, model_name, test_dicts, y_test)

    # Log metrics after retraining
    print("Generating log file for debugging...")
    log_metrics_to_postgres(model_name, drift_score, num_drifted_columns, share_missing_values)
    return new_model_name

# Define how columns are mapped for drift analysis
column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical_features,
    categorical_features=categorical_features
)

# Setup a report for drift analysis using Evidently AI
report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
])

@task
def prep_db():
    """
    Prepare the PostgreSQL database and table for logging drift metrics.

    Creates the database if it doesn't exist and initializes the metrics table.
    """
    with psycopg.connect(f"host={HOST} port=5432 dbname=postgres user={USER} password={PASSWORD}", autocommit=True) as conn:
        res = conn.execute(f"SELECT 1 FROM pg_database WHERE datname='{DBNAME}'") # type: ignore
        if len(res.fetchall()) == 0:
            conn.execute(f"CREATE DATABASE {DBNAME};") # type: ignore
            logging.info(f"Database {DBNAME} created.")
        else:
            logging.info(f"Database {DBNAME} already exists.")

    with psycopg.connect(f"host={HOST} port=5432 dbname={DBNAME} user={USER} password={PASSWORD}", autocommit=True) as conn:
        conn.execute(create_table_statement) # type: ignore
        logging.info("Table `admission_metrics` created.")

@task
def calculate_metrics_postgresql(curr, i, data, reference_data):
    """
    Calculate drift metrics for a batch of data and log them to PostgreSQL.

    Args:
        curr (psycopg2 cursor): The database cursor for executing SQL commands.
        i (int): The current batch index.
        data (pd.DataFrame): The raw data to analyze.
        reference_data (pd.DataFrame): The reference data to compare against.
    """
    global pipeline  # Use the global pipeline variable
    batch_size = len(data) // 1000
    start_idx = i * batch_size
    end_idx = start_idx + batch_size if i < 27 else len(data)

    current_data = data.iloc[start_idx:end_idx]

    # Convert each row to a dictionary and then predict
    current_dicts = current_data[numerical_features + categorical_features].fillna(0).to_dict(orient='records')
    
    # Optimize by using pd.concat
    predictions = pd.Series(pipeline.predict(current_dicts), index=current_data.index, name='prediction')
    current_data = pd.concat([current_data, predictions], axis=1)

    # Run the report comparing reference data with current data
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    curr.execute(
        "INSERT INTO admission_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)",
        (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values)
    )

    # Check for drift and trigger retraining if necessary
    if prediction_drift > 0.6:
        MODEL_NAME = retraing(prediction_drift, num_drifted_columns, share_missing_values)
        pipeline = fetch_model(model_name=MODEL_NAME, alias="Production")  # Update the global pipeline variable

@flow
def batch_monitoring_backfill(reference_path=None, raw_path=None):
    """
    Perform batch monitoring and backfill historical data for drift analysis.

    Args:
        reference_path (str): The path to the reference dataset.
        raw_path (str): The path to the raw dataset.
    """
    # Prepare reference and raw data
    reference_data, raw_data = prepare_reference_and_raw_data(
        path=DATASET_PATH,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        reference_path=reference_path,
        raw_path=raw_path
    )

    # Prepare the reference data for the report
    reference_dicts = prepare_dictionaries(reference_data, numerical_features, categorical_features)
    predictions = pd.Series(pipeline.predict(reference_dicts), index=reference_data.index, name='prediction')
    reference_data = pd.concat([reference_data, predictions], axis=1)

    # Database preparation
    prep_db()

    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    
    with psycopg.connect(f"host={HOST} port=5432 dbname={DBNAME} user={USER} password={PASSWORD}", autocommit=True) as conn:
        for i in range(0, 1000):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i, raw_data, reference_data)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("Data sent")

if __name__ == '__main__':
    batch_monitoring_backfill()

