"""Services for synchronizing local files with the configured object store."""

from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings
from src.storage.factory import create_object_store
from src.utils.logger import get_logger


logger = get_logger(__name__)


class StorageSyncService:
    """Upload local raw data and artifacts to the configured object store."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.object_store = create_object_store(settings)

    def upload_raw_data(self) -> list[str]:
        """Upload every file from data/raw to the raw object prefix."""
        if self.settings.storage_backend == "local":
            logger.info("STORAGE_BACKEND=local; upload para object store ignorado.")
            return []
        if not self.settings.raw_data_dir.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {self.settings.raw_data_dir}")

        uploaded: list[str] = []
        for path in self.settings.raw_data_dir.iterdir():
            if not path.is_file():
                continue
            key = self.settings.raw_object_key(path.name)
            self.object_store.upload_file(path, key)
            uploaded.append(key)
            logger.info("Arquivo enviado para %s: %s", self.object_store.describe(), key)
        return uploaded

    def upload_artifacts(self, history_run_dir: Path | None = None) -> list[str]:
        """Upload trained model artifacts to the object store."""
        if self.settings.storage_backend == "local":
            logger.info("STORAGE_BACKEND=local; upload de artefatos ignorado.")
            return []

        artifacts = [
            (self.settings.pipeline_path, self.settings.pipeline_object_key),
            (self.settings.metadata_path, self.settings.metadata_object_key),
            (
                self.settings.threshold_analysis_path,
                self.settings.artifact_object_key(self.settings.threshold_analysis_filename),
            ),
            (
                self.settings.leakage_report_path,
                self.settings.artifact_object_key(self.settings.leakage_report_filename),
            ),
        ]
        if self.settings.baseline_dir.exists():
            artifacts.extend(
                (
                    path,
                    self.settings.artifact_object_key(
                        path.relative_to(self.settings.artifacts_dir).as_posix()
                    ),
                )
                for path in self.settings.baseline_dir.iterdir()
                if path.is_file()
            )
        if history_run_dir is not None:
            artifacts.extend(
                (
                    path,
                    self.settings.artifact_object_key(
                        path.relative_to(self.settings.artifacts_dir).as_posix()
                    ),
                )
                for path in history_run_dir.rglob("*")
                if path.is_file()
            )
            if self.settings.training_history_index_path.exists():
                artifacts.append(
                    (
                        self.settings.training_history_index_path,
                        self.settings.artifact_object_key(
                            self.settings.training_history_index_path.relative_to(
                                self.settings.artifacts_dir
                            ).as_posix()
                        ),
                    )
                )
        uploaded: list[str] = []
        for local_path, key in artifacts:
            if not local_path.exists():
                continue
            self.object_store.upload_file(local_path, key)
            uploaded.append(key)
            logger.info("Artefato enviado para %s: %s", self.object_store.describe(), key)
        return uploaded

    def download_artifact(self, key: str, local_path: Path) -> bool:
        """Download one artifact when it exists in the configured object store."""
        if self.settings.storage_backend == "local":
            return local_path.exists()
        if not self.object_store.exists(key):
            return False
        self.object_store.download_file(key, local_path)
        logger.info("Artefato baixado de %s: %s", self.object_store.describe(), key)
        return True
