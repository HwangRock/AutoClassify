import boto3
import os

def download_model_from_s3(bucket_name, s3_key, local_path):
    if not os.path.exists(local_path):
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Model downloaded from S3 to {local_path}")
    else:
        print(f"Model already exists at {local_path}")
