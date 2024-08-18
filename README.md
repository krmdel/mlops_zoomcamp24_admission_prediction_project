# Admission Prediction Web Service

This project utilized the dataset provided by W. S. Hong et al. in their paper "Predicting hospital admission at emergency department triage using machine learning" ([Link to Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016)). The dataset is available at [GitHub Repository](https://github.com/yaleemmlc/admissionprediction.git).

## The Admission Prediction Web Service

The Admission Prediction Web Service is a Flask API that predicts the probability of a patient being admitted to the hospital based on the patient's health information. The web service is built using Flask and deployed on:

- **AWS Elastic Compute Cloud (EC2)** for computing
- **S3** for cloud storage
- **Relational Databases (RDS)** for managing databases
- **Elastic Container Registry (ECR)** for storing Docker images

The model is trained using the XGBoost algorithm and is stored in an S3 bucket via MLflow. The web service retrieves the model from the S3 bucket and uses it to make predictions. 

Streamlit is used to create a web application that allows users to interact with the web service. Users can upload a CSV file containing patient(s) health information and receive prediction(s) of the probability of the patient being admitted to the hospital, while storing the results in an S3 bucket.

## Monitoring

The monitoring of the web service is done using:

- **MLflow** to track the experiments and metrics
- **Grafana** to visualize the metrics
- **Adminer** to manage the database

If the prediction drift exceeds the threshold, the web service initiates a conditional workflow and logs the metrics/dashboards. The conditional workflow performs retraining of different logistic regression models, finds the best performing one, and further performs hyperparameter tuning for model registry. The best model is then deployed to the production environment.

## Deployment

The web service is deployed on AWS EC2, RDS, and S3.#   m l o p s _ z o o m c a m p 2 4 _ a d m i s s i o n _ p r e d i c t i o n _ p r o j e c t  
 