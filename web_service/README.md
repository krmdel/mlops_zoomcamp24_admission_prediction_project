**Admission Prediction Web Service**<br/>
<br/>
This repository contains a web service for predicting admissions using machine learning models. The service is built with Flask, and the models are managed with MLflow. The instructions below guide you through setting up the environment, building and running the Docker container, and deploying the Docker image to AWS Elastic Container Registry (ECR).<br/>
<br/>
**Table of Contents**<br/>
<br/>
**Prerequisites**<br/>
<br/>
**Environment Setup**<br/>
<br/>
**Building and Running Docker**<br/>
<br/>
**Publishing Docker Image to ECR**<br/>
<br/>
**Running Docker Image from ECR**<br/>
<br/>
**References<br/>
<br/>
**Prerequisites**<br/>
<br/>
Before you begin, ensure you have the following:<br/>
<br/>
Python 3.10 installed on your system.<br/>
<br/>
Docker installed on your system.<br/>
<br/>
AWS CLI configured with your credentials.<br/>
<br/>
AWS Elastic Container Registry (ECR) set up.<br/>
<br/>
**Environment Setup**<br/>
<br/>
**1.Create a Virtual Environment and Install Dependencies**<br/>
<br/>
Use pipenv to create a virtual environment and install the required dependencies:<br/>
<br/>
`pipenv install scikit-learn==1.5.1 xgboost==2.1.0 mlflow flask boto3 tqdm joblib python-dotenv gunicorn --python==3.10`
<br/>
**2.Activate the Virtual Environment**<br/>
<br/>
Enter the virtual environment created by pipenv:<br/>
<br/>
pipenv shell<br/>
<br/>
Building and Running Docker<br/>
<br/>
- Build the Docker Image:<br/>
<br/>
Build the Docker image for the web service:<br/>
<br/>
docker build -t admission_prediction_service:v1 .<br/>
<br/>
- Set Up AWS Environment Variables:<br/>
<br/>
Export your AWS credentials and region as environment variables:<br/>
<br/>
export AWS_ACCESS_KEY_ID=<your_aws_access_key_id><br/>
<br/>
export AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key><br/>
<br/>
export AWS_DEFAULT_REGION=<your_aws_region><br/>
<br/>
- Run the Docker Container:<br/>
<br/>
Run the Docker container locally, exposing the necessary ports and passing the AWS credentials:<br/>
<br/>
docker run -it --rm \<br/>
    -p 9696:9696 \<br/>
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \<br/>
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \<br/>
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \<br/>
    admission_prediction_service:v1<br/>
<br/>
Publishing Docker Image to ECR<br/>
<br/>
1.	Create a Repository on ECR:<br/>
<br/>
Create a new repository on AWS ECR for your Docker image:<br/>
<br/>
aws ecr create-repository --repository-name admission_prediction_service<br/>
<br/>
note down "repositoryUri" in the output to push the docker image as below<br/>
<br/>
2.	Login to ECR:<br/>
<br/>
Authenticate Docker to the ECR registry:<br/>
<br/>
$(aws ecr get-login --no-include-email)<br/>
<br/>
3.	Tag and Push the Docker Image to ECR:<br/>
<br/>
Tag the local Docker image and push it to the ECR repository:<br/>
<br/>
REMOTE_URI=ECR_URI ("repositoryUri")<br/>
<br/>
REMOTE_TAG="v1"  # version of the image<br/>
<br/>
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}<br/>
<br/>

LOCAL_IMAGE=admission_prediction_service:v1<br/>
<br/>
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}<br/>
<br/>
docker push ${REMOTE_IMAGE}<br/>
<br/>
Running Docker Image from ECR<br/>
<br/>
1.	Pull the Docker Image from ECR:<br/>
<br/>
Pull the Docker image from the ECR repository:<br/>
<br/>
docker pull $REMOTE_URI:v1<br/>
<br/>
2.	Run the Docker Container from ECR:<br/>
<br/>
Run the Docker container from the ECR image, exposing the necessary ports and passing the AWS credentials:<br/>
<br/>
docker run -it --rm -p 8501:8501 -p 9696:9696 \<br/>
   -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \<br/>
   -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \<br/>
   -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \<br/>
   $REMOTE_URI:v1<br/>
<br/>
3.	Run Streamlit Application:<br/>
<br/>
Start the Streamlit application by running:<br/>
<br/>
streamlit run app.py<br/>
<br/>
Once the above steps are completed, docker is up and running/Streamlit can access docker, Streamlit UI provides an easy-to-use and informative screen to both provide path of CSV file or upload it for model to make predictions (admission/discharge). The patient ID, patient’s health information and prediction are displayed in UI. Model can make multiple predictions on a CSV file including many patients. Dropdown menu shows each patient and by clicking each of them, the predictions and patient information can be accessed. After predictions are made, the results are uploaded to S3 bucket by creating a new folder and name the prediction file as JSON with date and patient ID to easy access.<br/>
<br/>
References<br/>
<br/>
Detailed guide on how to use AWS ECR for container management can be found here: <br/>
<br/>
https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/04-deployment/streaming<br/>
<br/>
https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html<br/>
<br/>


