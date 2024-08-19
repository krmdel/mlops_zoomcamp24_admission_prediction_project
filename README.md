![Admission decisions are crucial in emergency rooms](images\emergency.jpg)
**Admission Prediction Web Service**<br/>
<br/>
This project addresses the critical challenge of predicting hospital admissions at the time of emergency department (ED) triage by leveraging both patient history and triage information. Accurate prediction of hospital admission can significantly enhance patient care and resource allocation in healthcare settings.<br/>
<br/>
**Problem Overview**<br/>
<br/>
Emergency departments are often overwhelmed, making it crucial to accurately predict which patients are likely to require hospital admission. Traditional methods rely primarily on triage information, which may not capture the full complexity of a patient's health status. This project, inspired by the research conducted by W. S. Hong et al. in their paper "Predicting hospital admission at emergency department triage using machine learning", seeks to improve predictive performance by incorporating both triage data and comprehensive patient history.<br/>
<br/>
**Solution Overview**<br/>
<br/>
The Admission Prediction Web Service is a Flask API designed to predict the probability of a patient being admitted to the hospital using machine learning models trained on a rich dataset that includes both triage and historical patient data. This web service:<br/>
<br/>
**Model Training:** Utilizes the XGBoost algorithm, known for its robust performance, to train predictive models on a dataset that includes over 972 variables per patient visit, such as demographics, triage information, and past medical history.<br/>
<br/>
**Deployment:** The trained model is stored in an S3 bucket using MLflow for easy retrieval and deployment. The Flask API is deployed on AWS Elastic Compute Cloud (EC2) with support from AWS services like S3 for cloud storage, RDS for database management, and ECR for Docker image storage.<br/>
<br/>
**User Interaction:** A Streamlit-based web application allows users to interact with the service by uploading a CSV file containing patient health information. The application then provides predictions on the likelihood of hospital admission, which are also stored in an S3 bucket for further analysis.<br/>
<br/>
**Monitoring and Maintenance:** The web service includes comprehensive monitoring using MLflow to track experiments and metrics, Grafana for visualizing metrics, and Adminer for database management. If the model's performance drifts beyond a certain threshold, the service triggers a conditional workflow that retrains and tunes the model to ensure optimal performance. The best-performing model is automatically promoted to production.<br/>
<br/>
**Key Features**<br/>
<br/>
**- Enhanced Predictive Accuracy:** By integrating both triage and patient history data, the web service achieves higher predictive accuracy compared to models using triage data alone.<br/>
<br/>
**- Scalable Deployment:** Deployed on AWS infrastructure, ensuring scalability and reliability.<br/>
<br/>
**- User-Friendly Interface:** Streamlit-based interface makes it easy for healthcare professionals to upload data and receive predictions.<br/>
<br/>
**- Continuous Improvement:** Built-in monitoring and automated retraining ensure that the model remains accurate over time.<br/>
<br/>
This web service is a powerful tool for healthcare providers, enabling them to make more informed decisions at the point of triage and ultimately improving patient outcomes.<br/>
<br/>
Dataset Used for training/testing:https://drive.google.com/drive/folders/11SvGdC7R0bWmm8vWNP43z24wpU0tnA-S?usp=sharing<br/>
<br/>
