Admission Prediction Web Service<br/>
<br/>
This project was utilized the dataset provided by W. S. Hong et al. in their paper "Predicting hospital admission at emergency department triage using machine learning" (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016). The dataset is available at https://github.com/yaleemmlc/admissionprediction.git<br/>
<br/>
The Admission Prediction Web Service is a Flask API that predicts the probability of a patient being admitted to the hospital based on the patient's health information. The web service is built using Flask and deployed on AWS Elastic Compute Cloud (EC2) for computing, S3 for cloud storage, Relational Databases (RDS) for managing databases and Elastic Container Registry (ECR) for storing docker image. The model is trained using the XGBoost algorithm and is stored in an S3 bucket over MLflow. The web service retrieves the model from the S3 bucket and uses it to make predictions. Streamlit is used to create a web application that allows users to interact with the web service. User can upload a CSV file containing patient(s) health information and receive prediction(s) of the probability of the patient being admitted to the hospital, while storing the results in S3 bucket.<br/>
<br/>
The monitoring of the web service is done using MLflow, Grafana and Adminer. MLflow is used to track the experiments and metrics, Grafana is used to visualize the metrics and Adminer is used to manage the database. If the prediction drift exceeds the threshold, the web service initiates conditional workflow and log the metrics/dashboards. The conditional workflow performs retraining of different logistic regression models, finding the best performing one and further performs hyperparameter tuning for model registry. The best model is then deployed to the production environment. The web service is deployed on AWS EC2, RDS and S3.<br/>#   m l o p s _ z o o m c a m p 2 4 _ a d m i s s i o n _ p r e d i c t i o n _ p r o j e c t 
 
 #   m l o p s _ z o o m c a m p 2 4 _ a d m i s s i o n _ p r e d i c t i o n _ p r o j e c t 
 
 

Admission Prediction Web Service<br/>
<br/>
This project addresses the critical challenge of predicting hospital admissions at the time of emergency department (ED) triage by leveraging both patient history and triage information. Accurate prediction of hospital admission can significantly enhance patient care and resource allocation in healthcare settings.<br/>
<br/>
Problem Overview<br/>
<br/>
Emergency departments are often overwhelmed, making it crucial to accurately predict which patients are likely to require hospital admission. Traditional methods rely primarily on triage information, which may not capture the full complexity of a patient's health status. This project, inspired by the research conducted by W. S. Hong et al. in their paper "Predicting hospital admission at emergency department triage using machine learning", seeks to improve predictive performance by incorporating both triage data and comprehensive patient history.<br/>
<br/>
Solution Overview<br/>
<br/>
The Admission Prediction Web Service is a Flask API designed to predict the probability of a patient being admitted to the hospital using machine learning models trained on a rich dataset that includes both triage and historical patient data. This web service:<br/>
<br/>
Model Training: Utilizes the XGBoost algorithm, known for its robust performance, to train predictive models on a dataset that includes over 972 variables per patient visit, such as demographics, triage information, and past medical history.<br/>
<br/>
Deployment: The trained model is stored in an S3 bucket using MLflow for easy retrieval and deployment. The Flask API is deployed on AWS Elastic Compute Cloud (EC2) with support from AWS services like S3 for cloud storage, RDS for database management, and ECR for Docker image storage.<br/>
<br/>
User Interaction: A Streamlit-based web application allows users to interact with the service by uploading a CSV file containing patient health information. The application then provides predictions on the likelihood of hospital admission, which are also stored in an S3 bucket for further analysis.<br/>
<br/>
Monitoring and Maintenance: The web service includes comprehensive monitoring using MLflow to track experiments and metrics, Grafana for visualizing metrics, and Adminer for database management. If the model's performance drifts beyond a certain threshold, the service triggers a conditional workflow that retrains and tunes the model to ensure optimal performance. The best-performing model is automatically promoted to production.<br/>
<br/>
Key Features<br/>
<br/>
Enhanced Predictive Accuracy: By integrating both triage and patient history data, the web service achieves higher predictive accuracy compared to models using triage data alone.
- Scalable Deployment: Deployed on AWS infrastructure, ensuring scalability and reliability.<br/>
<br/>
- User-Friendly Interface: Streamlit-based interface makes it easy for healthcare professionals to upload data and receive predictions.<br/>
<br/>
- Continuous Improvement: Built-in monitoring and automated retraining ensure that the model remains accurate over time.<br/>
<br/>
This web service is a powerful tool for healthcare providers, enabling them to make more informed decisions at the point of triage and ultimately improving patient outcomes.<br/>
<br/>#   m l o p s _ z o o m c a m p 2 4 _ a d m i s s i o n _ p r e d i c t i o n _ p r o j e c t  
 