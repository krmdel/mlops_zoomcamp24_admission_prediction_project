from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from mlflow.exceptions import MlflowException
import os
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

def initialize():
    # Set the SQLite tracking URI
    # mlflow.set_tracking_uri("sqlite:////home/ubuntu/mlflow.db")
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")


    # Start MLflow experiment
    now = datetime.now().strftime("date_%d_%m_%Y_time_%H_%M_%S")
    experiment_name = f"compare_model_performance_{now}"
    mlflow.set_experiment(f"{experiment_name}")

    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    experiments = mlflow.search_experiments()

    print(experiments)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        print(f"Experiment '{experiment_name}' created successfully.")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    else:
        print(f"Experiment '{experiment_name}' could not be created.")

    return experiment_name