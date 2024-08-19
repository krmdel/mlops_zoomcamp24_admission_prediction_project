import os
import pandas as pd
from io import StringIO
import uuid
from tqdm import tqdm
from datetime import datetime
import boto3

from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# S3 bucket configuration: Retrieve the bucket name from environment variables
BUCKET_NAME = os.getenv('BUCKET_NAME')
s3_client = boto3.client('s3')

def save_predictions_to_s3(results, bucket_name=BUCKET_NAME):
    """
    Save the prediction results to an S3 bucket with a dynamic file name.

    Args:
        results (list of dict): The prediction results to save.
        bucket_name (str): The name of the S3 bucket to upload to.
    """
    # Convert the results list of dictionaries to a DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Get the current date and time to generate unique file names
    current_date = datetime.now().strftime("%Y_%m_%d")
    current_time = datetime.now().strftime("%H_%M_%S")

    # Generate a file name based on the number of predictions and the patient ID (if single prediction)
    if len(results) == 1:
        patient_id = results[0]['patient_id']
        file_name = f"admission_predictions_{patient_id}_{current_date}_{current_time}.csv"
    else:
        file_name = f"admission_predictions_batch_{len(results)}_{current_date}_{current_time}.csv"
    
    # Create a folder structure in S3 using the current date to organize files
    folder_name = f"admission_predictions/{current_date}"
    
    # Convert the DataFrame to a CSV format in memory using StringIO
    csv_buffer = StringIO()
    results_df.to_csv(csv_buffer, index=False)

    # Upload the CSV data to the specified S3 bucket and folder
    s3_client = boto3.client('s3')
    s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/{file_name}", Body=csv_buffer.getvalue())
    
    return f"s3://{bucket_name}/{folder_name}/{file_name}"

def df_prepare(path):
    """
    Prepare a DataFrame from a large CSV file by loading it in chunks.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        DataFrame: The prepared and cleaned DataFrame.
    """
        
    # Get the total size of the file for progress tracking
    total_size = os.path.getsize(path)
    
    # Use tqdm to display a progress bar as the file is read and processed
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
        # Read the file in chunks to handle large files efficiently
        df = pd.read_csv(path, chunksize=500000)
        # Concatenate chunks into a single DataFrame
        df = pd.concat([chunk for chunk in tqdm(df, total=total_size//1024, unit='chunks', leave=False)])
        # Update the progress bar to indicate completion
        pbar.update(total_size)
    
    # Clean and standardize column names by converting to lowercase and replacing spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # If the 'uid' column doesn't exist, create unique identifiers for each row
    if 'uid' not in df.columns:
        df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Clean up categorical columns by standardizing text (lowercase, removing spaces, commas, colons)
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace(',', '').str.replace(':', '')
    
    return df

def prepare_dictionaries(df):
    """
    Prepare the DataFrame for machine learning by separating features into numerical and categorical lists.

    Args:
        df (DataFrame): The input DataFrame containing patient data.

    Returns:
        list of dict: A list of dictionaries where each dictionary contains features for one patient.
        list of str: A list of unique identifiers (UIDs) corresponding to each patient.
    """

    # Define the numerical and categorical features expected in the data
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

    # Fill missing values in numerical features with 0, and drop rows with missing categorical features
    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features)
    df.loc[:, categorical_features] = df[categorical_features].astype(str)

    # Merge numerical and categorical features into a list of dictionaries, one per patient
    feature_dicts = []
    for _, row in df.iterrows():
        features = {**row[numerical_features].to_dict(), **row[categorical_features].to_dict()}
        feature_dicts.append(features)

    return feature_dicts, df['uid'].tolist() # Return both the feature dictionaries and their corresponding UIDs

def return_data(path):
    """
    Load a CSV file and prepare it for machine learning predictions.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        list of dict: A list of feature dictionaries for each patient.
        list of str: A list of unique identifiers (UIDs) corresponding to each patient.
    """
    df = df_prepare(path) # Prepare and clean the DataFrame from the CSV
    return prepare_dictionaries(df) # Prepare the dictionaries and return the data