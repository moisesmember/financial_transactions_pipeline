"""Factory for storage backends."""

from __future__ import annotations

from src.config.settings import Settings
from src.storage.local_store import LocalObjectStore
from src.storage.object_store import ObjectStore


def create_object_store(settings: Settings) -> ObjectStore:
    """Create the configured object storage backend."""
    if settings.storage_backend == "local":
        return LocalObjectStore(settings.project_root)
    if settings.storage_backend == "minio":
        from src.storage.minio_store import MinioObjectStore

        return MinioObjectStore(settings)
    raise ValueError("STORAGE_BACKEND deve ser 'local' ou 'minio'.")
