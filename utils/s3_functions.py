from typing import Any

import boto3

from config import config

s3 = boto3.session.Session().client(
    service_name=config.BUCKET_SERVICE,
    endpoint_url=config.BUCKET_HOST,
    aws_access_key_id=config.BUCKET_KEY_ID,
    aws_secret_access_key=config.BUCKET_KEY,
)


def upload_fileobj(fileobj: Any, s3_filepath: str):
    s3.upload_fileobj(fileobj, config.BUCKET_NAME, s3_filepath)


def upload_file(local_filepath: str, s3_filepath: str):
    s3.upload_file(local_filepath, config.BUCKET_NAME, s3_filepath)


def download_file(s3_filepath: str, local_filepath: str):
    s3.download_file(config.BUCKET_NAME, s3_filepath, local_filepath)


def write_object(filepath: str, content: Any):
    s3.put_object(Body=content, Bucket=config.BUCKET_NAME, Key=filepath)
