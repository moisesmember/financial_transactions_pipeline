"""Upload local raw dataset files to MinIO.

Run with STORAGE_BACKEND=minio configured in the environment.
"""

from src.config.settings import Settings
from src.storage.sync import StorageSyncService


def main() -> None:
    """Upload every file in data/raw to the configured MinIO bucket."""
    uploaded = StorageSyncService(Settings(storage_backend="minio")).upload_raw_data()
    print("Arquivos enviados:")
    for key in uploaded:
        print(f"- {key}")


if __name__ == "__main__":
    main()
