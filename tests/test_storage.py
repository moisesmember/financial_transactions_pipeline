"""Tests for local object storage integration."""

from __future__ import annotations

import json

import pandas as pd

from src.config.settings import Settings
from src.storage.factory import create_object_store
from src.storage.sync import StorageSyncService


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
