"""Tests for optional MLflow tracking."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import joblib
import pandas as pd

from src.config.settings import Settings
from src.models.mlflow_tracking import MlflowTrackingService


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment = None
        self.created_experiment = None
        self.started_run = None
        self.params = None
        self.metrics = None
        self.artifacts = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def get_experiment_by_name(self, name: str):
        self.experiment = name
        return None

    def create_experiment(self, name: str, artifact_location: str | None = None) -> str:
        self.created_experiment = {
            "name": name,
            "artifact_location": artifact_location,
        }
        return "experiment-1"

    def start_run(self, experiment_id: str, run_name: str, tags: dict[str, str]):
        self.started_run = {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "tags": tags,
        }
        return _FakeRun("mlflow-run-1")

    def log_params(self, params: dict[str, str]) -> None:
        self.params = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics

    def log_artifacts(self, local_dir: str, artifact_path: str) -> None:
        self.artifacts = {
            "local_dir": local_dir,
            "artifact_path": artifact_path,
        }


class _FakeMlflowModels:
    def __init__(self) -> None:
        self.input_example = None

    def infer_signature(self, input_example, prediction_example):
        self.input_example = input_example
        return "signature"


class _FakeMlflowSklearn:
    def __init__(self) -> None:
        self.logged_model = None

    def log_model(
        self,
        sk_model,
        name=None,
        signature=None,
        input_example=None,
        registered_model_name=None,
        serialization_format=None,
    ) -> None:
        self.logged_model = {
            "sk_model": sk_model,
            "name": name,
            "signature": signature,
            "input_example": input_example,
            "registered_model_name": registered_model_name,
            "serialization_format": serialization_format,
        }


def test_mlflow_tracking_logs_governed_run_without_real_server(tmp_path, monkeypatch) -> None:
    fake_mlflow = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    run_dir = tmp_path / "artifacts" / "history" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    settings = Settings(
        project_root=tmp_path,
        mlflow_tracking_enabled=True,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="fraud-tests",
        mlflow_artifact_location="s3://bucket/mlflow",
        mlflow_log_model=False,
    )
    metadata = {
        "run_id": "run-1",
        "status": "completed",
        "model_name": "logistic_regression",
        "model_params": {"max_iter": 1000, "class_weight": "balanced"},
        "model_selection": {"engine": "fixed", "trial_count": 0},
        "threshold": 0.2,
        "threshold_selection": {
            "strategy": "business_cost",
            "false_positive_cost": 1,
            "false_negative_cost": 25,
            "analysis_start": 0.1,
            "analysis_stop": 0.9,
            "analysis_step": 0.1,
        },
        "validation_metrics": {"precision": 0.5, "recall": 0.7, "pr_auc": 0.8},
        "test_metrics": {"precision": 0.4, "recall": 0.6, "pr_auc": 0.7},
        "out_of_time_metrics": {"precision": 0.3, "recall": 0.5, "pr_auc": 0.6},
        "operational_costs": {"validation": 10, "test": 20, "out_of_time": 30},
        "dataset": {
            "train_rows": 100,
            "validation_rows": 20,
            "test_rows": 20,
            "out_of_time_rows": 10,
            "train_positive_rate": 0.1,
        },
        "feature_set_version": "v1",
        "code_version": "abc",
        "dataset_version": "dataset",
        "training_max_rows": None,
        "strict_leakage_prevention": True,
        "geo_ablation_enabled": False,
        "baseline_decision": {"decision": "keep_candidate"},
        "experiment_fingerprint": "fingerprint",
    }

    mlflow_run_id = MlflowTrackingService(settings).log_completed_run(
        run_dir,
        metadata,
        {"status": "pass"},
    )

    assert mlflow_run_id == "mlflow-run-1"
    assert fake_mlflow.tracking_uri == "http://mlflow:5000"
    assert fake_mlflow.created_experiment == {
        "name": "fraud-tests",
        "artifact_location": "s3://bucket/mlflow",
    }
    assert fake_mlflow.started_run["run_name"] == "run-1"
    assert fake_mlflow.started_run["tags"]["training_run_id"] == "run-1"
    assert fake_mlflow.params["model_param_max_iter"] == "1000"
    assert fake_mlflow.metrics["selected_threshold"] == 0.2
    assert fake_mlflow.metrics["test_business_cost"] == 20.0
    assert fake_mlflow.artifacts["artifact_path"] == "governance"


def test_mlflow_model_logging_uses_cloudpickle_and_stable_input_schema(
    tmp_path,
) -> None:
    fake_mlflow = SimpleNamespace(
        models=_FakeMlflowModels(),
        sklearn=_FakeMlflowSklearn(),
    )
    run_dir = tmp_path / "history" / "run-1"
    run_dir.mkdir(parents=True)
    joblib.dump({"model": "fake"}, run_dir / "fraud_pipeline.joblib")
    settings = Settings(
        project_root=tmp_path,
        mlflow_tracking_enabled=True,
        mlflow_log_model=True,
    )

    MlflowTrackingService(settings)._log_model(
        fake_mlflow,
        run_dir,
        input_example=pd.DataFrame({"card_id": [1, 2], "amount": [10.0, 20.0]}),
        prediction_example=[0.1, 0.2],
    )

    assert fake_mlflow.sklearn.logged_model["name"] == "model"
    assert fake_mlflow.sklearn.logged_model["signature"] == "signature"
    assert fake_mlflow.sklearn.logged_model["serialization_format"] == "cloudpickle"
    assert str(fake_mlflow.models.input_example["card_id"].dtype) == "float64"
