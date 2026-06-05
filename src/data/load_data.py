"""Repository layer for reading raw fraud detection data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config.settings import Settings
from src.storage.factory import create_object_store
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RawDataRepository:
    """Read CSV and JSON files from the configured raw data directory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.raw_dir = settings.raw_data_dir
        self.object_store = create_object_store(settings)

    def _resolve_file(self, candidates: Iterable[str], required: bool = True) -> Path | str | None:
        """Find the first existing local file or object key from candidate names."""
        for name in candidates:
            if self.settings.storage_backend == "local":
                path = self.raw_dir / name
                if path.exists():
                    return path
            else:
                key = self.settings.raw_object_key(name)
                if self.object_store.exists(key):
                    return key
        if required:
            expected = ", ".join(candidates)
            location = self.raw_dir if self.settings.storage_backend == "local" else self.object_store.describe()
            raise FileNotFoundError(f"Nenhum arquivo encontrado em {location}: {expected}")
        return None

    def load_csv(self, candidates: Iterable[str], required: bool = True) -> pd.DataFrame:
        """Load a CSV using candidate filenames."""
        path = self._resolve_file(candidates, required=required)
        if path is None:
            return pd.DataFrame()
        logger.info("Carregando CSV de %s: %s", self.settings.storage_backend, path)
        if self.settings.storage_backend == "local":
            return pd.read_csv(path)
        return self.object_store.read_csv(str(path))

    def load_json(self, candidates: Iterable[str], required: bool = True) -> object:
        """Load a JSON file using candidate filenames."""
        path = self._resolve_file(candidates, required=required)
        if path is None:
            return {}
        logger.info("Carregando JSON de %s: %s", self.settings.storage_backend, path)
        if self.settings.storage_backend != "local":
            return self.object_store.read_json(str(path))
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)

    def load_transactions(self) -> pd.DataFrame:
        """Load transaction records."""
        return self.load_csv(self.settings.transaction_file_candidates)

    def load_cards(self) -> pd.DataFrame:
        """Load card records."""
        return self.load_csv(self.settings.cards_file_candidates, required=False)

    def load_users(self) -> pd.DataFrame:
        """Load user records."""
        return self.load_csv(self.settings.users_file_candidates, required=False)

    def load_mcc_codes(self) -> object:
        """Load merchant category code descriptions."""
        return self.load_json(self.settings.mcc_file_candidates, required=False)

    def load_labels(self) -> object:
        """Load fraud labels."""
        return self.load_json(self.settings.labels_file_candidates)

    def load_all(self) -> dict[str, object]:
        """Load all raw dataset components."""
        return {
            "transactions": self.load_transactions(),
            "cards": self.load_cards(),
            "users": self.load_users(),
            "mcc": self.load_mcc_codes(),
            "labels": self.load_labels(),
        }
