from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score
from mlflow.exceptions import MlflowException

# Initialize the MlflowClient to interact with the MLflow tracking server
client = MlflowClient()

def test_model(logged_model, test_dicts, y_test):
    """
    Load a machine learning model from MLflow and evaluate its accuracy on a test dataset.

    Args:
        logged_model (str): The URI of the model logged in MLflow.
        test_dicts (list of dict): The test data in the form of a list of dictionaries (features).
        y_test (list or array): The true labels for the test data.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    try:
        # Load the model using the logged URI from MLflow
        pipeline = mlflow.sklearn.load_model(logged_model)

        # Predict the labels for the test data
        y_test_pred = pipeline.predict(test_dicts)

        # Calculate the accuracy of the model
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Print the accuracy and the model's URI
        print(f"Model URI: {logged_model} - Accuracy: {test_accuracy}")

        return test_accuracy
    except MlflowException as e:
        # Handle exceptions if the model cannot be loaded or if there's an issue with MLflow
        print(f"Model URI: {logged_model} not found: {e}")
        return 0

def evaluate_models(model_name, test_dicts, y_test):
    """
    Evaluate all versions of a registered model in MLflow, rank them by accuracy, 
    and update their aliases accordingly.

    Args:
        model_name (str): The name of the registered model in MLflow.
        test_dicts (list of dict): The test data in the form of a list of dictionaries (features).
        y_test (list or array): The true labels for the test data.

    Returns:
        None
    """
    # Get the current date and time for logging and description purposes
    date = datetime.now().strftime("%d/%m/%Y - %H:%M:%S")

    # Retrieve all versions of the model with the specified name from MLflow
    versions = client.search_model_versions(f"name='{model_name}'")

    # Initialize a dictionary to store accuracies of each model version
    accuracies = {}

    # Iterate through each model version
    for version in versions:
        # Get the source URI of the current model version
        model_uri = version.source
        
        # Test the model and store its accuracy
        accuracy = test_model(logged_model=model_uri, test_dicts=test_dicts, y_test=y_test)
        accuracies[version.version] = accuracy

        # Print the URI of the tested model
        print(model_uri)

    # Sort the model versions by their accuracy in descending order
    sorted_versions = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)

    # Update model aliases based on their accuracy ranking
    for i, (version, accuracy) in enumerate(sorted_versions):
        if i == 0:
            # Assign 'Production' alias to the model with the highest accuracy
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
            # Assign 'Staging_X' alias to other models ranked by accuracy
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

    # Print out the accuracies of all the evaluated model versions
    print("Model Accuracies:", accuracies)

    # Print the version and accuracy of the best-performing model
    print(f"Best Model Version: {sorted_versions[0][0]} with Accuracy: {sorted_versions[0][1]}")

