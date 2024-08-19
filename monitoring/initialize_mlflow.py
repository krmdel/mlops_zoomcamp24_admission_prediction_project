from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from mlflow.exceptions import MlflowException
import os
from dotenv import load_dotenv

# Load environment variables from the .env file located in the parent directory of this script
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Retrieve the MLflow tracking server host from the environment variables
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

def initialize():
    """
    Initialize the MLflow tracking environment and create a new experiment.

    This function sets up the connection to the MLflow tracking server using the
    tracking URI specified in the environment variable. It creates a new experiment
    with a unique name based on the current date and time, and prints out the details
    of the created experiment.

    Returns:
        str: The name of the created experiment.
    """
    # Set the tracking URI for MLflow to connect to the remote server
    # Uncomment the following line to use a local SQLite database instead
    # mlflow.set_tracking_uri("sqlite:////home/ubuntu/mlflow.db")
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    # Create a unique experiment name using the current date and time
    now = datetime.now().strftime("date_%d_%m_%Y_time_%H_%M_%S")
    experiment_name = f"compare_model_performance_{now}"

    # Set the current experiment in MLflow to the newly created experiment
    mlflow.set_experiment(f"{experiment_name}")

    # Print the tracking URI to confirm connection to the correct server
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

    # Retrieve and print a list of all experiments from the tracking server
    experiments = mlflow.search_experiments()
    print(experiments)

    # Fetch the experiment by its name to confirm it was created
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Print details about the experiment if it was successfully created
    if experiment:
        print(f"Experiment '{experiment_name}' created successfully.")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
    else:
        # If the experiment was not created, output an error message
        print(f"Experiment '{experiment_name}' could not be created.")
    
    # Return the name of the experiment for use in further MLflow operations
    return experiment_name