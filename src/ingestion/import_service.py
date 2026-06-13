"""Application service for importing external datasets into object storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.settings import Settings
from src.ingestion.kaggle_source import KaggleDatasetSource
from src.ingestion.source import DatasetSource
from src.storage.factory import create_object_store
from src.storage.object_store import ObjectStore
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetImportResult:
    """Keys written to and skipped in the configured storage."""

    imported: tuple[str, ...]
    skipped: tuple[str, ...]


class DatasetImportService:
    """Coordinate a dataset source and a destination object store."""

    def __init__(
        self,
        settings: Settings,
        source: DatasetSource | None = None,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.settings = settings
        self.source = source or KaggleDatasetSource(settings.kaggle_dataset)
        self.object_store = object_store or create_object_store(settings)

    def import_data(self, overwrite: bool | None = None) -> DatasetImportResult:
        """Import Kaggle files, preserving existing objects unless overwrite is enabled."""
        should_overwrite = self.settings.kaggle_overwrite if overwrite is None else overwrite
        logger.info(
            "Importacao iniciada | origem=%s | destino=%s | overwrite=%s",
            self.source.describe(),
            self.object_store.describe(),
            should_overwrite,
        )

        if not should_overwrite:
            existing = self._existing_expected_keys()
            if len(existing) == len(self.settings.kaggle_expected_files):
                logger.info(
                    "Importacao ignorada | motivo=arquivos_existentes | arquivos=%d",
                    len(existing),
                )
                return DatasetImportResult(imported=(), skipped=tuple(existing))

        with TemporaryDirectory(prefix="fraud-kaggle-") as staging:
            staging_dir = Path(staging)
            downloaded_files = self.source.download(staging_dir)
            result = self._persist(downloaded_files, staging_dir, should_overwrite)

        logger.info(
            "Importacao concluida | importados=%d | ignorados=%d | destino=%s",
            len(result.imported),
            len(result.skipped),
            self.object_store.describe(),
        )
        return result

    def _persist(
        self,
        files: list[Path],
        staging_dir: Path,
        overwrite: bool,
    ) -> DatasetImportResult:
        imported: list[str] = []
        skipped: list[str] = []

        for path in files:
            relative_path = path.relative_to(staging_dir).as_posix()
            key = self.settings.raw_object_key(relative_path)
            if self.object_store.exists(key) and not overwrite:
                skipped.append(key)
                logger.info("Arquivo ignorado | motivo=existente | key=%s", key)
                continue

            self.object_store.upload_file(path, key)
            imported.append(key)
            logger.info("Arquivo persistido | key=%s | overwrite=%s", key, overwrite)

        return DatasetImportResult(imported=tuple(imported), skipped=tuple(skipped))

    def _existing_expected_keys(self) -> list[str]:
        keys = [self.settings.raw_object_key(filename) for filename in self.settings.kaggle_expected_files]
        return [key for key in keys if self.object_store.exists(key)]
