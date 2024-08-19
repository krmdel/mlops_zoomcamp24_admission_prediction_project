import time
from datetime import datetime
from dotenv import load_dotenv, set_key, dotenv_values
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost as xgb
import os
from dotenv import load_dotenv

# Load environment variables from a .env file located in the parent directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Retrieve the tracking server host URL from environment variables
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

# Initialize an Mlflow client to interact with the MLflow tracking server
client = MlflowClient()

# Update the variable
def update_env_variable(key, value, dotenv_path=env_path):
    """
    Update an environment variable both in the current session and in the .env file.

    Args:
        key (str): The name of the environment variable to update.
        value (str): The new value for the environment variable.
        dotenv_path (str): The path to the .env file. Defaults to the global env_path.

    Returns:
        None
    """
    # Update the variable in the environment
    os.environ[key] = value
    
    # Update the variable in the .env file
    set_key(dotenv_path, key, value)

def train_and_log_model(model_name, model, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path, numerical_features, categorical_features, target, register=False):
    """
    Train a machine learning model, log its performance metrics, and optionally register it with MLflow.

    Args:
        model_name (str): The name of the model.
        model (sklearn.base.BaseEstimator): The model to be trained.
        train_dicts (list of dict): Training data in dictionary format.
        y_train (list of str): Training labels.
        valid_dicts (list of dict): Validation data in dictionary format.
        y_valid (list of str): Validation labels.
        test_dicts (list of dict): Test data in dictionary format.
        y_test (list of str): Test labels.
        path (str): The file path to the dataset.
        numerical_features (list of str): List of numerical feature names.
        categorical_features (list of str): List of categorical feature names.
        target (str): The target variable name.
        register (bool): If True, register the model in MLflow. Default is False.

    Returns:
        tuple: A tuple containing validation accuracy, test accuracy, validation loss, and test loss.
    """
    print(f"Training {model_name}...")
    with mlflow.start_run() as run:
        start = time.time()
        mlflow.set_tag("model", model_name) # Set model name as a tag in MLflow
        mlflow.log_param("dataset_path", path) # Log dataset path as a parameter
        mlflow.log_param("numerical_features", numerical_features) # Log numerical features
        mlflow.log_param("categorical_features", categorical_features) # Log categorical features
        mlflow.log_param("target", target) # Log target variable
        model.fit(train_dicts, y_train) # Train the model

        # Perform cross-validation and log the mean accuracy
        accuracies = cross_val_score(estimator=model, X=train_dicts, y=y_train, cv=10)
        cross_val_mean = accuracies.mean()
        mlflow.log_metric("cross_val_mean_accuracy", cross_val_mean)

        # Validate the model and log validation accuracy and loss
        y_valid_pred = model.predict(valid_dicts)
        y_valid_prob = model.predict_proba(valid_dicts)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_loss = log_loss(y_valid, y_valid_prob)
        mlflow.log_metric("valid_accuracy", valid_accuracy)
        mlflow.log_metric("valid_loss", valid_loss)

        # Test the model and log test accuracy and loss
        y_test_pred = model.predict(test_dicts)
        y_test_prob = model.predict_proba(test_dicts)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_loss = log_loss(y_test, y_test_prob)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        # Log the trained model in MLflow
        mlflow.sklearn.log_model(model, f"{model_name}")

        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Loss: {test_loss}")

        end = time.time()

        # Print the time taken to train the model
        print(f"Time taken to train the model: {end - start:.2f} seconds")

        return valid_accuracy, test_accuracy, valid_loss, test_loss


def compare_model_performances(train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path, numerical_features, categorical_features, target):
    """
    Train and compare the performance of two machine learning models: Logistic Regression and XGBoost.

    Args:
        train_dicts (list of dict): Training data in dictionary format.
        y_train (list of str): Training labels.
        valid_dicts (list of dict): Validation data in dictionary format.
        y_valid (list of str): Validation labels.
        test_dicts (list of dict): Test data in dictionary format.
        y_test (list of str): Test labels.
        path (str): The file path to the dataset.
        numerical_features (list of str): List of numerical feature names.
        categorical_features (list of str): List of categorical feature names.
        target (str): The target variable name.

    Returns:
        str: The name of the best performing model.
    """
    # Define and create pipelines for Logistic Regression and XGBoost models
    logistic_regression = make_pipeline(
        DictVectorizer(),
        StandardScaler(with_mean=False),
        LogisticRegression(random_state=42, max_iter=100)
    )
    
    xgboost_model = make_pipeline(
        DictVectorizer(),
        StandardScaler(with_mean=False),
        XGBClassifier(objective='binary:logistic', eval_metric='logloss', seed=42)
    )
    
    # Train Logistic Regression model and log its performance
    lr_valid_acc, lr_test_acc, lr_valid_loss, lr_test_loss = train_and_log_model(
        "sklearn_logistic_regression", logistic_regression, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test,
        path, numerical_features, categorical_features, target
    )
    
    # Train XGBoost model and log its performance
    xgb_valid_acc, xgb_test_acc, xgb_valid_loss, xgb_test_loss = train_and_log_model(
        "xgboost_model", xgboost_model, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test,
        path, numerical_features, categorical_features, target
    )
    
    # Compare the performance of both models and select the best one
    if lr_test_loss > xgb_test_loss and xgb_test_acc > lr_test_acc:
        best_model = "xgboost_model"
    else:
        best_model = "sklearn_logistic_regression"
    
    print(f"Best model: {best_model}")
    return best_model

def compare_models_performance_and_select_best(model1, model2, test_dicts, y_test):
    """
    Compare the performance of the current production model with a newly trained model and select the best one.

    Args:
        model1 (str): Name of the first model (usually the new model).
        model2 (str): Name of the second model (usually the production model).
        test_dicts (list of dict): Test data in dictionary format.
        y_test (list of str): Test labels.

    Returns:
        str: The name of the model to be promoted to production.
    """
    # Initialize MLflow client
    client = mlflow.tracking.MlflowClient()
    
    # Fetch the current production model
    try:
        model_artifact1 = f"models:/{model1}@Production"
        prod_model1 = mlflow.sklearn.load_model(model_artifact1)
        print(f"Production model version {prod_model1} loaded for comparison.")
    except IndexError:
        print("No production model found. The newly trained model will be set as production.")
        return model1  # If no production model exists, promote the new model

    # Load the newly trained model
    model_artifact2 = f"models:/{model2}@Production"
    prod_model2 = mlflow.sklearn.load_model(model_artifact2)
    print(f"Production model version {prod_model2} loaded for comparison.")
    
    # Compare the test accuracy of the production model and the newly trained model
    y_test_pred1 = prod_model1.predict(test_dicts)
    y_test_pred2 = prod_model2.predict(test_dicts)

    test_accuracy1 = accuracy_score(y_test, y_test_pred1)
    test_accuracy2 = accuracy_score(y_test, y_test_pred2)

    print(f"Production Model - Accuracy: {test_accuracy1}")
    print(f"New Model - Accuracy: {test_accuracy2}")

    # Determine which model performs better and update the environment variable
    if (test_accuracy2 > test_accuracy1):
        print("Newly trained model performs better. It will be promoted to Production.")
        update_env_variable('MODEL_NAME', model2)
        return model2
    else:
        print("Production model performs better or is equivalent. It will remain in Production.")
        update_env_variable('MODEL_NAME', model1)
        return model1
    
    