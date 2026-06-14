"""Services for synchronizing local files with the configured object store."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree

from src.config.settings import Settings
from src.storage.factory import create_object_store
from src.storage.object_store import ObjectStore
from src.utils.logger import get_logger


logger = get_logger(__name__)


class StorageSyncService:
    """Synchronize managed local files with the configured object store."""

    def __init__(
        self,
        settings: Settings,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.settings = settings
        self.object_store = object_store or create_object_store(settings)

    def upload_raw_data(self) -> list[str]:
        """Upload every file from data/raw to the raw object prefix."""
        if self.settings.storage_backend == "local":
            logger.info("STORAGE_BACKEND=local; upload para object store ignorado.")
            return []
        if not self.settings.raw_data_dir.exists():
            raise FileNotFoundError(f"Diretorio nao encontrado: {self.settings.raw_data_dir}")

        uploaded: list[str] = []
        for path in self.settings.raw_data_dir.rglob("*"):
            if not path.is_file():
                continue
            key = self.settings.raw_object_key(
                path.relative_to(self.settings.raw_data_dir).as_posix()
            )
            self.object_store.upload_file(path, key)
            uploaded.append(key)
            logger.info("Arquivo enviado para %s: %s", self.object_store.describe(), key)
        self._verify_uploaded(uploaded)
        return uploaded

    def upload_artifacts(self, history_run_dir: Path | None = None) -> list[str]:
        """Upload the complete local artifact workspace to the object store."""
        if self.settings.storage_backend == "local":
            logger.info("STORAGE_BACKEND=local; upload de artefatos ignorado.")
            return []
        del history_run_dir
        if not self.settings.artifacts_dir.exists():
            return []

        artifacts = [
            (
                path,
                self.settings.artifact_object_key(
                    path.relative_to(self.settings.artifacts_dir).as_posix()
                ),
            )
            for path in self.settings.artifacts_dir.rglob("*")
            if path.is_file()
        ]
        uploaded: list[str] = []
        for local_path, key in artifacts:
            self.object_store.upload_file(local_path, key)
            uploaded.append(key)
            logger.info("Artefato enviado para %s: %s", self.object_store.describe(), key)
        self._verify_uploaded(uploaded)
        return uploaded

    def prepare_artifact_workspace(self) -> None:
        """Restore only remote state required to produce the next governed run."""
        if self.settings.storage_backend != "minio":
            return
        self.download_artifact(
            self.settings.artifact_object_key("history/runs.csv"),
            self.settings.training_history_index_path,
        )
        if self.settings.promote_baseline:
            self._download_remote_baseline()

    def purge_local_artifacts(self, uploaded: list[str] | None = None) -> None:
        """Remove local artifacts after every file is confirmed in MinIO."""
        if self.settings.storage_backend != "minio" or self.settings.keep_local_artifacts:
            return
        local_files = [path for path in self.settings.artifacts_dir.rglob("*") if path.is_file()]
        expected = {
            self.settings.artifact_object_key(
                path.relative_to(self.settings.artifacts_dir).as_posix()
            )
            for path in local_files
        }
        if uploaded is not None and not expected.issubset(set(uploaded)):
            raise RuntimeError("Limpeza local bloqueada: nem todos os artefatos foram enviados.")
        self._verify_uploaded(sorted(expected))
        self._remove_managed_directory(self.settings.artifacts_dir)
        logger.info("Artefatos locais removidos apos persistencia confirmada no MinIO.")

    def purge_local_raw_data(self, uploaded: list[str]) -> None:
        """Remove local raw data after every file is confirmed in MinIO."""
        if self.settings.storage_backend != "minio" or self.settings.keep_local_raw_data:
            return
        local_files = [path for path in self.settings.raw_data_dir.rglob("*") if path.is_file()]
        expected = {
            self.settings.raw_object_key(
                path.relative_to(self.settings.raw_data_dir).as_posix()
            )
            for path in local_files
        }
        if not expected.issubset(set(uploaded)):
            raise RuntimeError("Limpeza local bloqueada: nem todos os dados raw foram enviados.")
        self._verify_uploaded(sorted(expected))
        self._remove_managed_directory(self.settings.raw_data_dir)
        logger.info("Dados raw locais removidos apos persistencia confirmada no MinIO.")

    def download_artifact(self, key: str, local_path: Path) -> bool:
        """Download one artifact when it exists in the configured object store."""
        if self.settings.storage_backend == "local":
            return local_path.exists()
        if not self.object_store.exists(key):
            return False
        self.object_store.download_file(key, local_path)
        logger.info("Artefato baixado de %s: %s", self.object_store.describe(), key)
        return True

    def _download_remote_baseline(self) -> None:
        metadata_path = self.settings.baseline_dir / self.settings.baseline_metadata_filename
        metadata_key = self.settings.artifact_object_key(
            f"baseline/{self.settings.baseline_metadata_filename}"
        )
        if not self.download_artifact(metadata_key, metadata_path):
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline = metadata.get("baseline", {})
        filenames = {
            baseline.get("pipeline_file", self.settings.baseline_pipeline_filename),
            *baseline.get("reports", []),
        }
        for filename in filenames:
            if filename:
                self.download_artifact(
                    self.settings.artifact_object_key(f"baseline/{filename}"),
                    self.settings.baseline_dir / filename,
                )

    def _verify_uploaded(self, keys: list[str]) -> None:
        missing = [key for key in keys if not self.object_store.exists(key)]
        if missing:
            raise RuntimeError(
                "Objetos nao confirmados no storage: " + ", ".join(missing[:10])
            )

    def _remove_managed_directory(self, directory: Path) -> None:
        root = self.settings.project_root.resolve()
        target = directory.resolve()
        if target == root or root not in target.parents:
            raise RuntimeError(f"Limpeza recusada fora do workspace: {target}")
        if target.exists():
            rmtree(target)
