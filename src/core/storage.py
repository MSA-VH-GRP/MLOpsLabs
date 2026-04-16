"""MinIO / S3 client wrapper."""

import boto3
from botocore.client import Config

from src.core.config import settings


def get_s3_client():
    """Return a boto3 S3 client pointed at MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


def upload_bytes(bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    client = get_s3_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def download_bytes(bucket: str, key: str) -> bytes:
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()
