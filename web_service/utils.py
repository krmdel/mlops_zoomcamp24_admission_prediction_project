import os
import pandas as pd
from io import StringIO
import uuid
from tqdm import tqdm
from datetime import datetime
import boto3

from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# S3 bucket configuration
BUCKET_NAME = os.getenv('BUCKET_NAME')
s3_client = boto3.client('s3')

def save_predictions_to_s3(results, bucket_name=BUCKET_NAME):
    """
    Save the prediction results to an S3 bucket with a dynamic file name.

    Args:
        results (list of dict): The prediction results to save.
        bucket_name (str): The name of the S3 bucket to upload to.
    """
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Get the current date and time
    current_date = datetime.now().strftime("%Y_%m_%d")
    current_time = datetime.now().strftime("%H_%M_%S")

    # Determine the file name based on the number of patients
    if len(results) == 1:
        patient_id = results[0]['patient_id']
        file_name = f"admission_predictions_{patient_id}_{current_date}_{current_time}.csv"
    else:
        file_name = f"admission_predictions_batch_{len(results)}_{current_date}_{current_time}.csv"
    
    # Create a folder name based on the current date
    folder_name = f"admission_predictions/{current_date}"
    
    # Convert DataFrame to CSV and save to S3
    csv_buffer = StringIO()
    results_df.to_csv(csv_buffer, index=False)
    s3_client = boto3.client('s3')
    s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/{file_name}", Body=csv_buffer.getvalue())
    
    return f"s3://{bucket_name}/{folder_name}/{file_name}"

def df_prepare(path):
    total_size = os.path.getsize(path)
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
        df = pd.read_csv(path, chunksize=500000)
        df = pd.concat([chunk for chunk in tqdm(df, total=total_size//1024, unit='chunks', leave=False)])
        pbar.update(total_size)
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    if 'uid' not in df.columns:
        df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace(',', '').str.replace(':', '')
    
    return df

def prepare_dictionaries(df):
    numerical_features = [
        "age", "albumin_last", "albumin_max", "albumin_median", "albumin_min",
        "bloodculture,routine_count", "bloodculture,routine_last", "bloodculture,routine_npos",
        "cc_abdominalcramping", "cc_abdominaldistention", "cc_abdominalpain",
        "cc_abdominalpainpregnant", "cc_allergicreaction", "cc_bleeding/bruising",
        "cc_breastpain", "cc_chestpain", "cc_confusion", "cc_diarrhea", "cc_dizziness",
        "cc_fall>65", "cc_fever", "cc_hallucinations", "cc_headache", "cc_hypertension",
        "cc_hypotension", "cc_irregularheartbeat", "cc_nausea", "cc_overdose-accidental",
        "cc_overdose-intentional", "cc_poisoning", "cc_rapidheartrate", "cc_rectalbleeding",
        "cc_strokealert", "cc_unresponsive", "cc_urinaryretention", "cktotal_last", 
        "cktotal_max", "cktotal_median", "cktotal_min", "d-dimer_last", "d-dimer_max", 
        "d-dimer_median", "d-dimer_min", "esi", "n_admissions", "n_edvisits", "n_surgeries", 
        "platelets_last", "platelets_max", "platelets_median", "platelets_min", 
        "rbc_last", "rbc_max", "rbc_median", "rbc_min", "triage_vital_dbp", 
        "triage_vital_hr", "triage_vital_o2", "triage_vital_o2_device", "triage_vital_rr", 
        "triage_vital_sbp", "triage_vital_temp", "troponini(poc)_last", "troponini(poc)_max", 
        "troponini(poc)_median", "troponini(poc)_min", "troponint_last", "troponint_max", 
        "troponint_median", "troponint_min", "urineculture,routine_count", 
        "urineculture,routine_last", "urineculture,routine_npos", "viralinfect", 
        "wbc_last", "wbc_max", "wbc_median", "wbc_min"
    ]

    categorical_features = ['arrivalmode', 'gender', 'previousdispo']

    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features)
    df.loc[:, categorical_features] = df[categorical_features].astype(str)

    # Merge numerical and categorical features into a list of dictionaries
    feature_dicts = []
    for _, row in df.iterrows():
        features = {**row[numerical_features].to_dict(), **row[categorical_features].to_dict()}
        feature_dicts.append(features)

    return feature_dicts, df['uid'].tolist()  # Return all dictionaries and their corresponding patient IDs

def return_data(path):
    df = df_prepare(path)
    return prepare_dictionaries(df)
