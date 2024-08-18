Web service:

Creating virtual environment and installing dependencies:

pipenv install scikit-learn==1.5.1 xgboost==2.1.0 mlflow flask boto3 tqdm joblib python-dotenv gunicorn --python==3.10

pipenv shell

Building and running docker

docker build -t admission_prediction_service:v1 .

export AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=AWS_DEFAULT_REGION

docker run -it --rm \
    -p 9696:9696 \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
    admission_prediction_service:v1

Publishing docker image on ECR

aws ecr create-repository --repository-name admission_prediction_service

aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin 851725313484.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com

$(aws ecr get-login --no-include-email)

REMOTE_URI=ECR_URI
REMOTE_TAG="v1" # version of the image
REMOTE_IMAGE=${REMOTE_URI}:${REMOTE_TAG}

LOCAL_IMAGE=LOCAL_IMAGE_NAME
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}

Running docker on ECR

aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin ECR_URI

docker pull $REMOTE_URI:v1

docker run -it --rm -p 8501:8501 -p 9696:9696 \
   -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
   -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
   -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
   $REMOTE_URI:v1

streamlit run app.py