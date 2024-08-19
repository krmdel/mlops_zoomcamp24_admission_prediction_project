import pandas as pd
import uuid
import os
import boto3
from io import StringIO
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from config import numerical_features, categorical_features, target

from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Set up S3 bucket details from environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME")
FILE_KEY = os.getenv("FILE_KEY")


def read_data(BUCKET_NAME, FILE_KEY, chunk_size=50000):
    """
    Reads data from an S3 bucket in chunks and returns a DataFrame.

    Args:
        BUCKET_NAME (str): The name of the S3 bucket.
        FILE_KEY (str): The key (file path) in the S3 bucket.
        chunk_size (int): The size of each chunk to read from the S3 object. Default is 50,000 bytes.

    Returns:
        DataFrame: A Pandas DataFrame containing the concatenated data read from the S3 object.
        str: The S3 path to the file.
    """
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
    total_length = int(response['ContentLength'])
    
    # Initialize a list to hold chunks of CSV data
    csv_data = []
    with tqdm(total=total_length, unit='B', unit_scale=True, desc=FILE_KEY) as pbar:
        for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
            csv_data.append(chunk.decode('utf-8'))
            pbar.update(len(chunk))
    
    # Join all chunks into a single CSV string
    csv_data = ''.join(csv_data)
    
    # Convert the CSV string into a DataFrame
    data = StringIO(csv_data)
    df = pd.read_csv(data, chunksize=10000)
    
    # Return the DataFrame and the S3 path
    path = f"s3://{BUCKET_NAME}/{FILE_KEY}"  
    return df, path

def df_prepare(path=None):
    """
    Prepares the DataFrame for further processing by cleaning and standardizing it.

    Args:
        path (str): The file path to the CSV file. If None, data is read from S3 using the BUCKET_NAME and FILE_KEY.

    Returns:
        DataFrame: A cleaned and standardized Pandas DataFrame.
        str: The path to the processed file.
    """
    if path:
        total_size = os.path.getsize(path)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
            df = pd.read_csv(path, chunksize=500000)
            df = pd.concat([chunk for chunk in tqdm(df, total=total_size//1024, unit='chunks', leave=False)])
            pbar.update(total_size)
    else:
        df, path = read_data(BUCKET_NAME, FILE_KEY)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Add a unique identifier (UID) if not already present
    if 'uid' not in df.columns:
        df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Clean and standardize categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace(',', '').str.replace(':', '')
    
    return df, path

def prepare_dictionaries(df, numerical_features, categorical_features):
    """
    Converts a DataFrame into a list of dictionaries with the specified features, ready for machine learning processing.

    Args:
        df (DataFrame): The DataFrame to process.
        numerical_features (list of str): A list of numerical feature column names.
        categorical_features (list of str): A list of categorical feature column names.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a data point, with keys being feature names.
    """
    # Fill missing numerical data with 0 and drop rows with missing categorical features
    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features)
    
    # Convert categorical columns to string and prepare the dictionary list
    df.loc[:, categorical_features] = df[categorical_features].astype(str)
    return df[numerical_features + categorical_features].to_dict(orient='records')


def split_dataset(numerical_features, categorical_features, target, path=None):
    """
    Splits the dataset into training, validation, and test sets, and prepares them for machine learning.

    Args:
        numerical_features (list of str): A list of numerical feature column names.
        categorical_features (list of str): A list of categorical feature column names.
        target (str): The target column name for classification.
        path (str): The file path to the CSV file. If None, data is read from S3 using the BUCKET_NAME and FILE_KEY.

    Returns:
        tuple: A tuple containing training, validation, and test dictionaries, their corresponding target labels, and the original DataFrames.
    """
    # Prepare the DataFrame
    df, path = df_prepare(path)

    # Handle missing values and drop rows with missing target values
    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features + [target])
    
    # Split the data into training, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)
    
    # Prepare dictionaries for machine learning models
    train_dicts = prepare_dictionaries(train_df, numerical_features, categorical_features)
    valid_dicts = prepare_dictionaries(valid_df, numerical_features, categorical_features)
    test_dicts = prepare_dictionaries(test_df, numerical_features, categorical_features)
    
    # Convert target labels to binary format
    y_train = train_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    y_valid = valid_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    y_test = test_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    
    print("Data processing done...")
    
    return train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, train_df, valid_df, test_df, path

def prepare_reference_and_raw_data(path, numerical_features, categorical_features, reference_path=None, raw_path=None):
    """
    Prepares and returns reference and raw data for comparison or validation purposes.

    Args:
        path (str): The file path to the original dataset.
        numerical_features (list of str): A list of numerical feature column names.
        categorical_features (list of str): A list of categorical feature column names.
        reference_path (str): Optional path to a reference dataset CSV file.
        raw_path (str): Optional path to a raw dataset CSV file.

    Returns:
        DataFrame: A reference DataFrame containing a sample of data points.
        DataFrame: A raw DataFrame containing another sample of data points.
    """
    if reference_path and raw_path:
        # Read the reference and raw datasets if paths are provided
        reference_df = pd.read_csv(reference_path)
        raw_df = pd.read_csv(raw_path)
        print(f"Reference data read from {reference_path}")
        print(f"Raw data read from {raw_path}")
    else:
        # Generate reference and raw datasets from the original data
        df, _ = df_prepare(path)

        # Ensure the dataset has the required structure
        df[numerical_features] = df[numerical_features].fillna(0)
        df = df.dropna(subset=categorical_features)

        # Sample 1000 patients randomly for both reference and raw datasets
        reference_df = df.sample(n=1000, random_state=78)
        raw_df = df.drop(reference_df.index).sample(n=1000, random_state=78)

        print("Reference and raw datasets generated from the original dataset")

    return reference_df, raw_df