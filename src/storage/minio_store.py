"""MinIO/S3-backed object store implementation."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from minio import Minio
from minio.error import S3Error

from src.config.settings import Settings
from src.storage.object_store import ObjectStore


class MinioObjectStore(ObjectStore):
    """ObjectStore implementation for MinIO-compatible buckets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket

    def ensure_bucket(self) -> None:
        """Create the configured bucket when it does not exist."""
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def exists(self, key: str) -> bool:
        """Return whether an object exists in MinIO."""
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}:
                return False
            raise

    def read_csv(self, key: str) -> pd.DataFrame:
        """Read a CSV object into a dataframe."""
        response = self.client.get_object(self.bucket, key)
        try:
            return pd.read_csv(BytesIO(response.read()))
        finally:
            response.close()
            response.release_conn()

    def read_json(self, key: str) -> Any:
        """Read a JSON object."""
        response = self.client.get_object(self.bucket, key)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to MinIO."""
        self.ensure_bucket()
        self.client.fput_object(self.bucket, key, str(local_path))

    def download_file(self, key: str, local_path: Path) -> None:
        """Download an object from MinIO to a local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, key, str(local_path))

    def describe(self) -> str:
        """Return backend description."""
        scheme = "https" if self.settings.minio_secure else "http"
        return f"minio:{scheme}://{self.settings.minio_endpoint}/{self.bucket}"
