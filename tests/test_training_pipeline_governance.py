"""End-to-end governance artifact test with a small synthetic dataset."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config.settings import Settings
from src.data.load_data import RawDataRepository
from src.pipelines.training_pipeline import TrainingPipeline


def test_training_pipeline_generates_governance_artifacts(tmp_path, monkeypatch) -> None:
    row_count = 240
    transactions = pd.DataFrame(
        {
            "id": range(row_count),
            "date": pd.date_range("2020-01-01", periods=row_count, freq="h"),
            "card_id": [index % 12 for index in range(row_count)],
            "client_id": [index % 20 for index in range(row_count)],
            "merchant_id": [index % 15 for index in range(row_count)],
            "merchant_city": ["A" if index % 3 else "B" for index in range(row_count)],
            "merchant_state": ["SP" if index % 3 else "RJ" for index in range(row_count)],
            "mcc": [5812 if index % 2 else 5411 for index in range(row_count)],
            "amount": [500.0 if index % 10 == 0 else 20.0 + index % 7 for index in range(row_count)],
        }
    )
    labels = {
        "target": {
            str(index): ("Yes" if index % 10 == 0 else "No")
            for index in range(row_count)
        }
    }
    raw = {
        "transactions": transactions,
        "cards": pd.DataFrame(),
        "users": pd.DataFrame(),
        "mcc": {"5812": "Restaurants", "5411": "Grocery"},
        "labels": labels,
    }
    monkeypatch.setattr(RawDataRepository, "load_all", lambda self: raw)
    settings = Settings(
        project_root=tmp_path,
        database_tracking_enabled=False,
        training_max_rows=None,
        threshold_analysis_start=0.10,
        threshold_analysis_stop=0.90,
        threshold_analysis_step=0.20,
        categorical_min_frequency=2,
        optuna_model_candidates=("logistic_regression",),
        optuna_trials=2,
        optuna_timeout_seconds=60,
        external_benchmarks_enabled=False,
    )

    result = TrainingPipeline(settings).run()

    metadata = joblib.load(settings.metadata_path)
    decision = json.loads(
        settings.artifact_path(settings.baseline_decision_filename).read_text(encoding="utf-8")
    )
    assert result.out_of_time_metrics["pr_auc"] >= 0
    assert metadata["dataset"]["out_of_time_rows"] > 0
    assert metadata["dataset_version"]
    assert metadata["model_selection"]["engine"] == "optuna"
    assert metadata["model_selection"]["trial_count"] == 2
    assert metadata["model_name"] == "logistic_regression"
    assert decision["decision"] in {"promote", "keep_candidate", "reject"}
    for filename in settings.governance_artifact_filenames:
        if filename == settings.geo_ablation_filename:
            continue
        assert (result.history_run_dir / filename).exists()
    manifest = json.loads(
        (result.history_run_dir / settings.manifest_filename).read_text(encoding="utf-8")
    )
    assert any(item["filename"] == "metadata.json" for item in manifest["artifacts"])
