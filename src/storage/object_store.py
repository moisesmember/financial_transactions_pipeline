"""Object storage interfaces used by data and artifact repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ObjectStore(ABC):
    """Common interface for local and object-storage backends."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether an object exists."""

    @abstractmethod
    def read_csv(self, key: str) -> pd.DataFrame:
        """Read a CSV object into a dataframe."""

    @abstractmethod
    def read_json(self, key: str) -> Any:
        """Read a JSON object."""

    @abstractmethod
    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to the object key."""

    @abstractmethod
    def download_file(self, key: str, local_path: Path) -> None:
        """Download an object to a local file."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable backend description."""
