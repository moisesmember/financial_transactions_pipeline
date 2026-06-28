"""Upload existing local data and artifacts to MinIO, then apply retention policy."""

from __future__ import annotations

from src.config.settings import Settings
from src.storage.sync import StorageSyncService


def main() -> None:
    """Migrate managed local files to MinIO with verification before cleanup."""
    settings = Settings()
    if settings.storage_backend != "minio":
        raise ValueError("Configure STORAGE_BACKEND=minio antes de executar a migracao.")

    sync = StorageSyncService(settings)
    raw_objects = sync.upload_raw_data() if settings.raw_data_dir.exists() else []
    artifact_objects = sync.upload_artifacts() if settings.artifacts_dir.exists() else []
    sync.purge_local_raw_data(raw_objects)
    sync.purge_local_artifacts(artifact_objects)
    print(
        "Migracao concluida | "
        f"raw={len(raw_objects)} | artifacts={len(artifact_objects)} | "
        f"destino={sync.object_store.describe()}"
    )


if __name__ == "__main__":
    main()
