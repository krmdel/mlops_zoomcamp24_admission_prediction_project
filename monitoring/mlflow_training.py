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

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

client = MlflowClient()

# Update the variable
def update_env_variable(key, value, dotenv_path=env_path):
    # Update the variable in the environment
    os.environ[key] = value
    
    # Update the variable in the .env file
    set_key(dotenv_path, key, value)

def train_and_log_model(model_name, model, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path, numerical_features, categorical_features, target, register=False):
    print(f"Training {model_name}...")
    with mlflow.start_run() as run:
        start = time.time()
        mlflow.set_tag("model", model_name)
        mlflow.log_param("dataset_path", path)
        mlflow.log_param("numerical_features", numerical_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.log_param("target", target)
        model.fit(train_dicts, y_train)

        accuracies = cross_val_score(estimator=model, X=train_dicts, y=y_train, cv=10)
        cross_val_mean = accuracies.mean()
        mlflow.log_metric("cross_val_mean_accuracy", cross_val_mean)

        y_valid_pred = model.predict(valid_dicts)
        y_valid_prob = model.predict_proba(valid_dicts)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_loss = log_loss(y_valid, y_valid_prob)
        mlflow.log_metric("valid_accuracy", valid_accuracy)
        mlflow.log_metric("valid_loss", valid_loss)

        y_test_pred = model.predict(test_dicts)
        y_test_prob = model.predict_proba(test_dicts)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_loss = log_loss(y_test, y_test_prob)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.sklearn.log_model(model, f"{model_name}")

        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Loss: {test_loss}")

        end = time.time()

        return valid_accuracy, test_accuracy, valid_loss, test_loss


def compare_model_performances(train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path, numerical_features, categorical_features, target):
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
    
    lr_valid_acc, lr_test_acc, lr_valid_loss, lr_test_loss = train_and_log_model(
        "sklearn_logistic_regression", logistic_regression, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test,
        path, numerical_features, categorical_features, target
    )
    
    xgb_valid_acc, xgb_test_acc, xgb_valid_loss, xgb_test_loss = train_and_log_model(
        "xgboost_model", xgboost_model, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test,
        path, numerical_features, categorical_features, target
    )
    
    if lr_test_loss > xgb_test_loss and xgb_test_acc > lr_test_acc:
        best_model = "xgboost_model"
    else:
        best_model = "sklearn_logistic_regression"
    
    print(f"Best model: {best_model}")
    return best_model

def compare_models_performance_and_select_best(model1, model2, test_dicts, y_test):
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
    
    y_test_pred1 = prod_model1.predict(test_dicts)
    y_test_pred2 = prod_model2.predict(test_dicts)

    test_accuracy1 = accuracy_score(y_test, y_test_pred1)
    test_accuracy2 = accuracy_score(y_test, y_test_pred2)

    print(f"Production Model - Accuracy: {test_accuracy1}")
    print(f"New Model - Accuracy: {test_accuracy2}")

    if (test_accuracy2 > test_accuracy1):
        print("Newly trained model performs better. It will be promoted to Production.")
        update_env_variable('MODEL_NAME', model2)
        return model2
    else:
        print("Production model performs better or is equivalent. It will remain in Production.")
        update_env_variable('MODEL_NAME', model1)
        return model1
    
    