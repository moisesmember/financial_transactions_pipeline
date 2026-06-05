"""Filesystem-backed object store implementation."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
from typing import Any

import pandas as pd

from src.storage.object_store import ObjectStore


class LocalObjectStore(ObjectStore):
    """ObjectStore implementation that maps keys to local files."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _path(self, key: str) -> Path:
        """Resolve an object key under the base directory."""
        return self.base_dir / key

    def exists(self, key: str) -> bool:
        """Return whether a local file exists."""
        return self._path(key).exists()

    def read_csv(self, key: str) -> pd.DataFrame:
        """Read a local CSV file into a dataframe."""
        return pd.read_csv(self._path(key))

    def read_json(self, key: str) -> Any:
        """Read a local JSON file."""
        with self._path(key).open("r", encoding="utf-8") as file:
            return json.load(file)

    def upload_file(self, local_path: Path, key: str) -> None:
        """Copy a local file into the target key."""
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != destination.resolve():
            copyfile(local_path, destination)

    def download_file(self, key: str, local_path: Path) -> None:
        """Copy a local key to another local path."""
        source = self._path(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != local_path.resolve():
            copyfile(source, local_path)

    def describe(self) -> str:
        """Return backend description."""
        return f"local:{self.base_dir}"
