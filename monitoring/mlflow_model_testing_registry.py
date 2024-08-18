from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from mlflow.exceptions import MlflowException

client = MlflowClient()

def test_model(logged_model, test_dicts, y_test):
    try:
        pipeline = mlflow.sklearn.load_model(logged_model)
        y_test_pred = pipeline.predict(test_dicts)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Model URI: {logged_model} - Accuracy: {test_accuracy}")
        return test_accuracy
    except MlflowException as e:
        print(f"Model URI: {logged_model} not found: {e}")
        return 0

def evaluate_models(model_name, test_dicts, y_test):
    # Get the current date
    date = datetime.now().strftime("%d/%m/%Y - %H:%M:%S")

    # List all model versions and test their accuracy
    versions = client.search_model_versions(f"name='{model_name}'")
    accuracies = {}
    for version in versions:
        # Get the source URI of the model version
        model_uri = version.source
        
        # Test the model and store its accuracy
        accuracy = test_model(logged_model=model_uri, test_dicts=test_dicts, y_test=y_test)
        accuracies[version.version] = accuracy
        print(model_uri)

    # Sort the versions by accuracy in descending order
    sorted_versions = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)

    # Update aliases based on sorted accuracy
    for i, (version, accuracy) in enumerate(sorted_versions):
        if i == 0:
            client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=version
            )
            client.update_model_version(
                name=model_name,
                version=version,
                description=f"The model version {version} was assigned alias 'Production' on {date} with accuracy {accuracy}"
            )
            print(f"Set alias 'Production' for model version {version}")
        else:
            staging_alias = f"Staging_{i}"
            client.set_registered_model_alias(
                name=model_name,
                alias=staging_alias,
                version=version
            )
            client.update_model_version(
                name=model_name,
                version=version,
                description=f"The model version {version} was assigned alias '{staging_alias}' on {date} with accuracy {accuracy}"
            )
            print(f"Set alias '{staging_alias}' for model version {version}")

    # Print final accuracies and versions
    print("Model Accuracies:", accuracies)
    print(f"Best Model Version: {sorted_versions[0][0]} with Accuracy: {sorted_versions[0][1]}")

