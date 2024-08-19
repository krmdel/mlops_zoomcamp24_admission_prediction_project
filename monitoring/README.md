**Monitoring of the model performance using MLflow Experiment Tracking with Docker and Grafana**<br/>
<br/>
This repository contains scripts and instructions for setting up an MLflow tracking server on AWS, along with Grafana and Adminer for monitoring and database management. Follow the steps below to set up your environment and start tracking your machine learning experiments.<br/>
<br/>
**Table of Contents**<br/>
<br/>
•	**Prerequisites**<br/>
<br/>
•	**Environment Setup**<br/>
<br/>
•	**Step-by-Step Instructions**<br/>
<br/>
•	**References**<br/>
<br/>
**Prerequisites**<br/>
<br/>
Before you begin, ensure you have the following:<br/>
<br/>
•	An AWS EC2 instance running a compatible Linux distribution.<br/>
<br/>
•	Docker and Docker Compose installed on your EC2 instance.<br/>
<br/>
•	PostgreSQL database setup on AWS RDS.<br/>
<br/>
•	S3 bucket created for storing MLflow artifacts.<br/>
<br/>
**Environment Setup**<br/>
<br/>
**1.Clone the Repository**<br/>
<br/>
`git clone https://github.com/krmdel/mlops_zoomcamp24_admission_prediction_project.git`<br/>
<br/>
`cd mlops_zoomcamp24_admission_prediction_project`<br/>
<br/>
**2.Install Required Packages**<br/>
<br/>
Ensure that Docker and Docker Compose are installed on your EC2 instance. If not, you can install them using the following commands:<br/>
<br/>
Docker:<br/>
<br/>
`sudo apt-get update`<br/>
<br/>
`sudo apt-get install docker.io`<br/>
<br/>
Docker Compose:<br/>
<br/>
`sudo apt-get install docker-compose`<br/>
<br/>
**3.Set Up Environment Variables**<br/>
<br/>
Export the necessary environment variables required for MLflow to interact with your PostgreSQL database and S3 bucket:<br/>
<br/>
`export DB_USER=<your_db_user>`<br/>
<br/>
`export DB_PASSWORD=<your_db_password>`<br/>
<br/>
`export DB_ENDPOINT=<your_db_endpoint>`<br/>
<br/>
`export S3_BUCKET_NAME=<your_s3_bucket_name>`<br/>
<br/>
**Step-by-Step Instructions**<br/>
<br/>
**1.Initiate MLflow Tracking Server**<br/>
<br/>
Start the MLflow server with PostgreSQL as the backend store and S3 as the artifact storage:<br/>
<br/>
`mlflow server -h 0.0.0.0 -p 5000 \`<br/>
<br/>
    `--backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@${DB_ENDPOINT}:5432/DB_NAME \`<br/>
<br/>
    `--default-artifact-root s3://${S3_BUCKET_NAME}`<br/>
<br/>
•	Replace DB_NAME with your PostgreSQL database name from RDS.<br/>
<br/>
•	Replace EC2_ENDPOINT with the public IP or DNS of your EC2 instance.<br/>
<br/>
MLflow UI: Access the MLflow UI at http://EC2_ENDPOINT:5050<br/>
<br/>
**2.Initiate Grafana and Adminer**<br/>
<br/>
Use Docker Compose to start Grafana for monitoring and Adminer for database management:<br/>
<br/>
`cd ./monitoring`<br/>
<br/>
Update RDS endpoint in config file: .\monitoring\config\grafana_dashboards.yaml<br/>
<br/>
`docker-compose up --build`<br/>
<br/>
Grafana: Access Grafana at http://EC2_ENDPOINT:3000<br/>
<br/>
Adminer: Access Adminer at http://EC2_ENDPOINT:8080<br/>
<br/>
**3.Login to Grafana and Adminer**<br/>
<br/>
Grafana: Username and password are set “admin” as default<br/>
<br/>
Adminer: You should have below credentials from RDS for logging in:<br/>
<br/>
**System:** PostgreSQL<br/>
<br/>
**Server:** DB_ENDPOINT<br/>
<br/>
**Username:** DB_USER<br/>
<br/>
**Password:** DB_PASSWORD<br/>
<br/>
**Database:** DB_NAME<br/>
<br/>
**4.Running monitoring script**<br/>
<br/>
In monitoring directory, run `python admission_metrics_calculation.py`<br/>
<br/>
Once the script is run, it fetches the latest production model from MLflow. Since we do not have a time series patient data, therefore, the script generates raw and reference data by randomly sample from dataset and simulates as if data is being received throughout the future time points. However, if any other reference and raw data is available, the path of csv files can be given for performing predictions and monitoring.<br/>
<br/>
The script monitors the prediction drift over time to decide whether model’s performance degrades. The recent threshold was set as 0.6. If the threshold is exceeded, it calls retraining function to initiate conditional workflow and logs all the model training and performance comparisons. Workflow involves first training and performance comparison of pre-defined models (sklearn’s logistic regression and XGBoost models) and logging to MLflow. Afterwards, once the best performing model with default hyperparameters is determined, further hyperparameter search and model registry for fine-tuning is performed on MLflow (for simplicity, evaluation step was given as 2 (the registered, "Production" and "Staging_1", and logged models can be found under "mlflow_artifacts" directory) and step number can be increased in hyperparameter tuning.py script). Various different model architecture can be added to model zoo in the future.<br/>
<br/>
After training and registering the best performing model, this model is compared against the model already in production. If the performance improvement is obtained, the deployed model is replaced with newly trained model and monitoring is kept carried out.<br/>
<br/>
Model registry and assigning the best performing model for "Production" on MLflow<br/>
<br/>
![Model registry and assigning the best performing model for "Production"](https://github.com/krmdel/mlops_zoomcamp24_admission_prediction_project/blob/main/Images/mlflow.png?raw=true)<br/>
<br/>
The stored artifacts of logged and registered XGBoost model in "Production" on S3 bucket. The screenshot of S3 bucket for trained models (sklearn logistic regression and XGBoost) of performance comparison as well as registered "Production" and "Staging" models were added to "Images" directory. The artifacts can be also found under "artifact" directory for testing purposes<br/>
<br/>
![The stored artifacts of logged and registered XGBoost model in "Production" on S3 bucket](https://github.com/krmdel/mlops_zoomcamp24_admission_prediction_project/blob/main/Images/mlflow_artifacts_s3_production_model.png?raw=true)<br/>
<br/>
Model monitoring: showing a case where prediction drift exceeded the threshold of 0.6 which triggered conditional workflow for retraining model and logging<br/>
<br/>
![Model monitoring: showing a case where prediction drift exceeded the threshold of 0.6 which triggered conditional workflow for retraining model and logging](https://github.com/krmdel/mlops_zoomcamp24_admission_prediction_project/blob/main/Images/grafana.png?raw=true)<br/>
<br/>
**References**<br/>
<br/>
Detailed instructions on setting up MLflow on AWS can be found here: https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md<br/>
<br/>

