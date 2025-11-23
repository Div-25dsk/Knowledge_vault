import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
import uuid
import os

TEMP_DIR = 'temp'

def save_temp_file(uploaded_file):
   os.makedirs(TEMP_DIR, exist_ok=True)

   uploaded_file.file.seek(0)  
    
   _,ext = os.path.splitext(uploaded_file.filename)

   temp_filename = f"{uuid.uuid4()}{ext}"

   temp_path = os.path.join(TEMP_DIR,temp_filename)

   with open(temp_path,"wb") as f:
      f.write(uploaded_file.file.read())

   return temp_path


BUCKET_NAME = "knowledge-vault-div"
REGION = "ap-south-1"

def upload_file_to_s3(file_path,s3_folder ='uploads'):
    s3 = boto3.client("s3",region_name = REGION)
    _,ext = os.path.splitext(file_path)
    file_key = f"{s3_folder}/{uuid.uuid4()}{ext}"

    try:
      s3.upload_file(file_path,BUCKET_NAME,file_key)
      file_url = f"https://{BUCKET_NAME}.s3.{REGION}.amazonaws.com/{file_key}"
      return file_url
    
    except NoCredentialsError:
      return "ERROR:AWS CREDENDITAILS NOT FOUND."
    
    except ClientError as e:
      return f"Error: {e}"



