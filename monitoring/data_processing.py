import pandas as pd
import uuid
import os
import boto3
from io import StringIO
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from config import numerical_features, categorical_features, target

from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

BUCKET_NAME = os.getenv("BUCKET_NAME")
FILE_KEY = os.getenv("FILE_KEY")


def read_data(BUCKET_NAME, FILE_KEY, chunk_size=50000):
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
    total_length = int(response['ContentLength'])
    
    csv_data = []
    with tqdm(total=total_length, unit='B', unit_scale=True, desc=FILE_KEY) as pbar:
        for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
            csv_data.append(chunk.decode('utf-8'))
            pbar.update(len(chunk))
    
    csv_data = ''.join(csv_data)
    
    data = StringIO(csv_data)
    df = pd.read_csv(data, chunksize=10000)
    
    path = f"s3://{BUCKET_NAME}/{FILE_KEY}"  
    
    return df, path


def df_prepare(path=None):
    if path:
        total_size = os.path.getsize(path)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(path)) as pbar:
            df = pd.read_csv(path, chunksize=500000)
            df = pd.concat([chunk for chunk in tqdm(df, total=total_size//1024, unit='chunks', leave=False)])
            pbar.update(total_size)
    else:
        df, path = read_data(BUCKET_NAME, FILE_KEY)
    
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    if 'uid' not in df.columns:
        df['uid'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_').str.replace(',', '').str.replace(':', '')
    
    return df, path


def prepare_dictionaries(df, numerical_features, categorical_features):
    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features)
    df.loc[:, categorical_features] = df[categorical_features].astype(str)
    return df[numerical_features + categorical_features].to_dict(orient='records')


def split_dataset(numerical_features, categorical_features, target, path=None):
    df, path = df_prepare(path)
    df[numerical_features] = df[numerical_features].fillna(0)
    df = df.dropna(subset=categorical_features + [target])
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.333, random_state=42)
    
    train_dicts = prepare_dictionaries(train_df, numerical_features, categorical_features)
    valid_dicts = prepare_dictionaries(valid_df, numerical_features, categorical_features)
    test_dicts = prepare_dictionaries(test_df, numerical_features, categorical_features)
    
    y_train = train_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    y_valid = valid_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    y_test = test_df[target].apply(lambda x: 1 if x == 'admit' else 0).values.astype(int)
    print("Data processing done...")
    
    return train_dicts, y_train, valid_dicts, y_valid, test_dicts, y_test, train_df, valid_df, test_df, path

def prepare_reference_and_raw_data(path, numerical_features, categorical_features, reference_path=None, raw_path=None):
    if reference_path and raw_path:
        # If paths are provided, read the CSV files
        reference_df = pd.read_csv(reference_path)
        raw_df = pd.read_csv(raw_path)
        print(f"Reference data read from {reference_path}")
        print(f"Raw data read from {raw_path}")
    else:
        # Prepare the dataset from the original path
        df, _ = df_prepare(path)  # Only capture the DataFrame, ignore the path

        # Ensure the dataset has the required structure
        df[numerical_features] = df[numerical_features].fillna(0)
        df = df.dropna(subset=categorical_features)

        # Sample 1000 patients randomly for reference and raw datasets
        reference_df = df.sample(n=1000, random_state=78)
        raw_df = df.drop(reference_df.index).sample(n=1000, random_state=78)

        print("Reference and raw datasets generated from the original dataset")

    return reference_df, raw_df