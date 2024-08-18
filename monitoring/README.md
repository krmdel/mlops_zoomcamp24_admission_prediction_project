Initiate MLfow tracking server

Plese refer to the following link for more information:
https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md

export DB_USER=DB_USER
export DB_PASSWORD=DB_PASSWORD
export DB_ENDPOINT=DB_ENDPOINT
export S3_BUCKET_NAME=S3_BUCKET_NAME

mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://S3_BUCKET_NAME

MLflow at http://EC2_ENDPOINT:5050

Initiate Grafana and Adminer

docker-compose up --build

Access Grafana at http://EC2_ENDPOINT:3000
Adminer at http://EC2_ENDPOINT:8080