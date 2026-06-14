"""Tests for baseline promotion and leakage audit."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.split_data import DataSplits
from src.models.baseline import BaselineRegistry
from src.models.leakage_audit import LeakageAuditService
from src.models.training_history import TrainingHistoryRegistry
from src.storage.postgres_training_history import PostgresTrainingHistoryRepository


def test_baseline_registry_requires_explicit_overwrite(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.artifacts_dir.mkdir(parents=True)
    settings.pipeline_path.write_bytes(b"model")
    registry = BaselineRegistry(settings)

    metadata_path = registry.promote({"model_name": "test"}, audit_status="pass")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["baseline"]["audit_status"] == "pass"
    assert (settings.baseline_dir / settings.baseline_pipeline_filename).read_bytes() == b"model"
    with pytest.raises(FileExistsError):
        registry.promote({"model_name": "new"})


def test_leakage_audit_flags_snapshot_and_high_auc(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "transaction_id": [str(index) for index in range(12)],
            "amount": range(12),
            "card_on_dark_web": ["No"] * 12,
            "is_fraud": [0, 1] * 6,
        }
    )
    splits = DataSplits(
        train=frame.iloc[:6],
        validation=frame.iloc[6:9],
        test=frame.iloc[9:],
        time_column="date",
    )
    preprocessor = SimpleNamespace(
        transformers_=[
            ("numeric", object(), ["amount"]),
            ("categorical", object(), ["card_on_dark_web"]),
            ("drop", "drop", ["transaction_id", "date"]),
        ]
    )
    pipeline = SimpleNamespace(named_steps={"preprocessor": preprocessor})

    report = LeakageAuditService(settings).build_report(
        splits,
        pipeline,
        validation_metrics={"roc_auc": 0.997},
        test_metrics={"roc_auc": 0.996},
        selected_threshold=0.20,
    )

    assert report["status"] == "warning"
    assert report["checks"]["temporal_order_valid"] is True
    assert report["risk_columns"]["snapshot"] == ["card_on_dark_web"]
    assert report["checks"]["high_roc_auc_warning"] is True


def test_training_history_creates_immutable_run_and_comparison_index(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    settings.artifacts_dir.mkdir(parents=True)
    settings.pipeline_path.write_bytes(b"pipeline")
    settings.metadata_path.write_bytes(b"metadata")
    settings.threshold_analysis_path.write_text("split,threshold\nvalidation,0.2\n", encoding="utf-8")
    settings.leakage_report_path.write_text('{"status": "warning"}', encoding="utf-8")
    metadata = {
        "model_name": "logistic_regression",
        "threshold": 0.2,
        "training_max_rows": 1000,
        "strict_leakage_prevention": True,
        "validation_metrics": {
            "precision": 0.3,
            "recall": 0.8,
            "f1": 0.43,
            "fbeta": 0.6,
            "pr_auc": 0.5,
            "roc_auc": 0.9,
            "tp": 8,
            "fp": 18,
            "fn": 2,
        },
        "test_metrics": {
            "precision": 0.4,
            "recall": 0.9,
            "f1": 0.55,
            "fbeta": 0.7,
            "pr_auc": 0.6,
            "roc_auc": 0.92,
            "tp": 9,
            "fp": 13,
            "fn": 1,
        },
        "threshold_selection": {
            "strategy": "business_cost",
            "false_positive_cost": 1,
            "false_negative_cost": 25,
        },
        "operational_costs": {"validation": 68, "test": 38},
        "dataset": {
            "train_rows": 700,
            "validation_rows": 150,
            "test_rows": 150,
            "train_positive_rate": 0.01,
            "validation_positive_rate": 0.02,
            "test_positive_rate": 0.02,
        },
    }
    registry = TrainingHistoryRegistry(settings)
    run_id = registry.new_run_id("logistic_regression")
    now = pd.Timestamp("2026-06-13T20:00:00Z").to_pydatetime()

    run_dir = registry.record(
        run_id,
        metadata,
        {"status": "warning", "warnings": ["high auc"]},
        started_at=now,
        completed_at=now,
    )

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / settings.pipeline_filename).read_bytes() == b"pipeline"
    index = pd.read_csv(settings.training_history_index_path)
    assert index.loc[0, "run_id"] == run_id
    assert index.loc[0, "test_business_cost"] == 38
    with pytest.raises(FileExistsError):
        registry.record(
            run_id,
            metadata,
            {"status": "warning", "warnings": []},
            started_at=now,
            completed_at=now,
        )


def test_postgres_history_persistence_can_be_disabled(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, database_tracking_enabled=False)

    persisted = PostgresTrainingHistoryRepository(settings).persist_if_available(tmp_path)

    assert persisted is False


def test_database_url_is_built_from_postgres_settings(tmp_path) -> None:
    settings = Settings(
        project_root=tmp_path,
        database_url_override=None,
        postgres_host="db",
        postgres_port=5433,
        postgres_database="tracking",
        postgres_user="user@example",
        postgres_password="secret value",
    )

    assert settings.database_url == (
        "postgresql+psycopg://user%40example:secret+value@db:5433/tracking"
    )
