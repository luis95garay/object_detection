import argparse
import boto3
from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser()
parser.add_argument('--bucket_name', type=str, default="elvis-s3-mlflow", help='s3 bucket name')
parser.add_argument('--dataset_path', type=str, default="dataset/Yolov7Dataset.zip", help='location of the dataset inside the bucket')
parser.add_argument('--save_dir', type=str, default="/content/Yolov7Dataset.zip", help='where to save the file')
args = parser.parse_args()

# Load variables from the .env file
load_dotenv("credentials.env")

# Access the credentials using os.getenv()
s3 = boto3.client('s3')

s3.download_file(args.bucket_name, args.dataset_path, args.save_dir)

