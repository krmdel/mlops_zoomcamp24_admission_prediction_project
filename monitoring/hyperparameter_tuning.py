from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from datetime import datetime
import time

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

from mlflow_training import train_and_log_model
from config import numerical_features, categorical_features, target

import os
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

client = MlflowClient()

def objective(space, best_model, model_name, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path):
    with mlflow.start_run() as run:
        mlflow.set_tag("model", best_model)
        mlflow.log_params(space)

        mlflow.log_param("dataset_path", path)
        mlflow.log_param("numerical_features", numerical_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.log_param("target", target)

        if best_model == "xgboost_model":
            pipeline = make_pipeline(
                DictVectorizer(),
                StandardScaler(with_mean=False),
                xgb.XGBClassifier(
                    n_estimators=int(space['n_estimators']),
                    max_depth=int(space['max_depth']),
                    learning_rate=space['learning_rate'],
                    gamma=space['gamma'],
                    min_child_weight=space['min_child_weight'],
                    subsample=space['subsample'],
                    colsample_bytree=space['colsample_bytree'],
                    objective='binary:logistic',
                    eval_metric='logloss',
                    seed=42
                )
            )
        else:
            pipeline = make_pipeline(
                DictVectorizer(),
                StandardScaler(with_mean=False),
                LogisticRegression(
                    C=space['C'],
                    penalty=space['penalty'],
                    solver='lbfgs',
                    max_iter=int(space['max_iter']),
                    random_state=42
                )
            )
        
        # Fit the pipeline on the training data
        pipeline.fit(train_dicts, y_train)
        
        # Validation accuracy and loss
        y_valid_pred = pipeline.predict(valid_dicts)
        y_valid_prob = pipeline.predict_proba(valid_dicts)       
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_loss = log_loss(y_valid, y_valid_prob)
        mlflow.log_metric("valid_accuracy", valid_accuracy)
        mlflow.log_metric("valid_loss", valid_loss)

        # Test accuracy and loss
        y_test_pred = pipeline.predict(test_dicts)
        y_test_prob = pipeline.predict_proba(test_dicts)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_loss = log_loss(y_test, y_test_prob)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        # Log the entire pipeline as a model
        mlflow.sklearn.log_model(pipeline, f"{model_name}")

        # Register the model under the same registry name
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{model_name}",
            name=model_name
        )
        
        # Wait for the model version to be created
        latest_version = registered_model.version
        
        # Introduce a short delay to ensure the alias is properly assigned
        time.sleep(5)
        
        # Tag the version with additional information
        date = datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
        client.set_model_version_tag(
            name=model_name,
            version=latest_version,
            key="registered_date",
            value=date
        )

        return {'loss': test_loss, 'status': STATUS_OK}

def hyperparameter_tuning(best_model, model_name, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path):
    if best_model == "xgboost_model":
        space = {
            'max_depth': hp.choice('max_depth', range(5, 30, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
        }
    else:
        space = {
            'C': hp.loguniform('C', -4, 4),
            'penalty': hp.choice('penalty', ['l2']),
            'max_iter': hp.choice('max_iter', range(100, 500, 50))
        }

    trials = Trials()
    best_params = fmin(
        fn=lambda params: objective(params, best_model, model_name, train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, path),
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )

    print(f"Best hyperparameters for {best_model}: {best_params}")