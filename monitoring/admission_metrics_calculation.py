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
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    model_artifact = f"models:/{model_name}@{alias}"
    pipeline = mlflow.sklearn.load_model(model_artifact)
    print(f"The {model_name} model from {alias} is loaded...")
    return pipeline

# Initialize the model pipeline globally
pipeline = fetch_model()

def checked_features(data):
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
    vectorized_features = pipeline.named_steps['dictvectorizer'].transform([features])
    scaled_features = pipeline.named_steps['standardscaler'].transform(vectorized_features)
    pred = pipeline.named_steps['xgbclassifier'].predict(scaled_features)
    admission = "admit" if pred == 1 else "discharge"
    return admission

begin = datetime.datetime.now()

def log_metrics_to_postgres(model_name, drift_score, num_drifted_columns, share_missing_values):
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
    print("\n")
    print("THRESHOLD FOR PREDICTION EXCEEDED!!!\n")
    print("CONDITIONAL WORKFLOW INITIATED!!!\n")

    # Prepare dataset
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

    # # Hyperparameter tuning
    print("Best model is fine tuned...")
    model_name = f"admission_prediction_{best_model}_{date.today()}"
    hyperparameter_tuning(best_model, model_name, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path)

    # Evaluate models
    print("Model is tested and registered...")
    evaluate_models(model_name, test_dicts, y_test)

    # Evaluate tuned model against former production model
    new_model_name = compare_models_performance_and_select_best(MODEL_NAME, model_name, test_dicts, y_test)

    # Log metrics after retraining
    print("Generating log file for debugging...")
    log_metrics_to_postgres(model_name, drift_score, num_drifted_columns, share_missing_values)
    return new_model_name

column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=numerical_features,
    categorical_features=categorical_features
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
])

@task
def prep_db():
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

