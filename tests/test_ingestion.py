"""Unit tests for Kaggle dataset ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.ingestion.import_service import DatasetImportService
from src.ingestion.kaggle_source import KaggleDatasetSource
from src.ingestion.source import DatasetSource
from src.storage.object_store import ObjectStore


class FakeDatasetSource(DatasetSource):
    """Create deterministic files without accessing the network."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.download_calls = 0

    def describe(self) -> str:
        return "fake:dataset"

    def download(self, destination: Path) -> list[Path]:
        self.download_calls += 1
        downloaded: list[Path] = []
        for filename, content in self.files.items():
            path = destination / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            downloaded.append(path)
        return downloaded


class InMemoryObjectStore(ObjectStore):
    """ObjectStore test double used to represent a remote backend."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.objects

    def read_csv(self, key: str) -> pd.DataFrame:
        raise NotImplementedError

    def read_json(self, key: str) -> Any:
        return json.loads(self.objects[key].decode("utf-8"))

    def upload_file(self, local_path: Path, key: str) -> None:
        self.objects[key] = local_path.read_bytes()

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.write_bytes(self.objects[key])

    def describe(self) -> str:
        return "minio:test/dataset"


def _dataset_files(content: str = "original") -> dict[str, str]:
    return {
        "transactions_data.csv": f"id,value\n1,{content}\n",
        "cards_data.csv": "id\n1\n",
        "users_data.csv": "id\n1\n",
        "mcc_codes.json": "{}",
        "train_fraud_labels.json": "{}",
    }


def test_imports_kaggle_files_into_local_storage(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    source = FakeDatasetSource(_dataset_files())

    result = DatasetImportService(settings, source=source).import_data()

    assert len(result.imported) == 5
    assert result.skipped == ()
    assert (tmp_path / "data" / "raw" / "transactions_data.csv").exists()


def test_repeated_import_does_nothing_when_overwrite_is_disabled(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    initial_source = FakeDatasetSource(_dataset_files("original"))
    DatasetImportService(settings, source=initial_source).import_data()
    replacement_source = FakeDatasetSource(_dataset_files("replacement"))

    result = DatasetImportService(settings, source=replacement_source).import_data(overwrite=False)

    transaction_file = tmp_path / "data" / "raw" / "transactions_data.csv"
    assert result.imported == ()
    assert len(result.skipped) == 5
    assert replacement_source.download_calls == 0
    assert "original" in transaction_file.read_text(encoding="utf-8")


def test_repeated_import_replaces_files_when_overwrite_is_enabled(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    DatasetImportService(settings, source=FakeDatasetSource(_dataset_files("original"))).import_data()
    replacement_source = FakeDatasetSource(_dataset_files("replacement"))

    result = DatasetImportService(settings, source=replacement_source).import_data(overwrite=True)

    transaction_file = tmp_path / "data" / "raw" / "transactions_data.csv"
    assert len(result.imported) == 5
    assert result.skipped == ()
    assert replacement_source.download_calls == 1
    assert "replacement" in transaction_file.read_text(encoding="utf-8")


def test_import_uses_same_overwrite_policy_for_minio_backend(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, storage_backend="minio")
    store = InMemoryObjectStore()
    key = settings.raw_object_key("transactions_data.csv")
    store.objects[key] = b"existing"
    source = FakeDatasetSource({"transactions_data.csv": "replacement"})

    skipped = DatasetImportService(settings, source=source, object_store=store).import_data(overwrite=False)
    replaced = DatasetImportService(settings, source=source, object_store=store).import_data(overwrite=True)

    assert skipped.imported == ()
    assert skipped.skipped == (key,)
    assert replaced.imported == (key,)
    assert store.objects[key] == b"replacement"


def test_kaggle_source_builds_official_cli_download_command(tmp_path) -> None:
    commands: list[list[str]] = []

    def runner(command) -> None:
        commands.append(list(command))
        destination = Path(command[command.index("--path") + 1])
        (destination / "transactions_data.csv").write_text("id\n1\n", encoding="utf-8")

    source = KaggleDatasetSource(
        "computingvictor/transactions-fraud-datasets",
        command_runner=runner,
    )

    files = source.download(tmp_path)

    assert commands == [
        [
            "kaggle",
            "datasets",
            "download",
            "computingvictor/transactions-fraud-datasets",
            "--path",
            str(tmp_path),
            "--unzip",
            "--quiet",
        ]
    ]
    assert files == [tmp_path / "transactions_data.csv"]
