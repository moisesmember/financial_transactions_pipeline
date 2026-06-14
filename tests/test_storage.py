"""Tests for local object storage integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.storage.factory import create_object_store
from src.storage.object_store import ObjectStore
from src.storage.sync import StorageSyncService


class MemoryObjectStore(ObjectStore):
    """Small object store fake used to validate synchronization behavior."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.objects

    def read_csv(self, key: str) -> pd.DataFrame:
        raise NotImplementedError

    def read_json(self, key: str) -> Any:
        raise NotImplementedError

    def upload_file(self, local_path: Path, key: str) -> None:
        self.objects[key] = local_path.read_bytes()

    def download_file(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.objects[key])

    def describe(self) -> str:
        return "memory:test"


def test_local_object_store_reads_csv_and_json(tmp_path) -> None:
    """Local object store should read files by object-like keys."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sample.csv").write_text("a,b\n1,x\n", encoding="utf-8")
    (raw_dir / "sample.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    settings = Settings(project_root=tmp_path)
    store = create_object_store(settings)

    frame = store.read_csv("data/raw/sample.csv")
    payload = store.read_json("data/raw/sample.json")

    assert frame.to_dict(orient="records") == [{"a": 1, "b": "x"}]
    assert payload == {"x": 1}


def test_local_sync_service_is_noop_for_upload(tmp_path) -> None:
    """Sync upload should be a no-op when STORAGE_BACKEND is local."""
    settings = Settings(project_root=tmp_path, raw_data_dir=tmp_path / "data" / "raw")
    settings.raw_data_dir.mkdir(parents=True)
    (settings.raw_data_dir / "transactions_data.csv").write_text("id,amount\n1,10\n", encoding="utf-8")

    uploaded = StorageSyncService(settings).upload_raw_data()

    assert uploaded == []


def test_minio_only_uploads_complete_artifact_tree_before_cleanup(tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    history_file = artifacts_dir / "history" / "run-1" / "metadata.json"
    benchmark_file = artifacts_dir / "external_benchmarks" / "run-1" / "model.bin"
    history_file.parent.mkdir(parents=True)
    benchmark_file.parent.mkdir(parents=True)
    history_file.write_text('{"run_id": "run-1"}', encoding="utf-8")
    benchmark_file.write_bytes(b"model")
    settings = Settings(
        project_root=tmp_path,
        storage_backend="minio",
        keep_local_artifacts=False,
        keep_local_raw_data=False,
    )
    store = MemoryObjectStore()
    sync = StorageSyncService(settings, object_store=store)

    uploaded = sync.upload_artifacts()
    sync.purge_local_artifacts(uploaded)

    assert set(uploaded) == {
        "artifacts/history/run-1/metadata.json",
        "artifacts/external_benchmarks/run-1/model.bin",
    }
    assert not artifacts_dir.exists()
    assert store.objects["artifacts/external_benchmarks/run-1/model.bin"] == b"model"


def test_local_backend_rejects_remote_only_retention(tmp_path) -> None:
    try:
        Settings(
            project_root=tmp_path,
            storage_backend="local",
            keep_local_artifacts=False,
        )
    except ValueError as exc:
        assert "KEEP_LOCAL_ARTIFACTS" in str(exc)
    else:
        raise AssertionError("Configuracao local sem retencao deveria ser rejeitada.")
