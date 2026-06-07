"""MinIO / S3-compatible object store for the original uploaded files.

- keeps the source file so we can re-process or show the cited page
- S3 API via boto3; endpoint + creds from env (.env)
- bucket auto-created on first use
"""

from __future__ import annotations

import os

import boto3
from botocore.exceptions import ClientError


class ObjectStore:
    """Put/get original files in a single bucket."""

    def __init__(self, bucket: str | None = None) -> None:
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "documents")
        self.client = boto3.client(
            "s3",
            endpoint_url=f"http://{endpoint}",
            aws_access_key_id=os.getenv("MINIO_ROOT_USER", "accel"),
            aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "accel_local_pw"),
            region_name="us-east-1",  # arbitrary; MinIO ignores it
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)

    def put(self, key: str, data: bytes) -> None:
        """Store bytes under `key` (e.g. the document_id/filename)."""
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        """Read back the bytes stored under `key`."""
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
