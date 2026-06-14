"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from time import sleep
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import (
    app,
    get_model_run_fact_repository,
    get_training_job_manager,
)
from src.api.schemas import TrainingRequest
from src.api.training_service import (
    TrainingAlreadyRunningError,
    TrainingJobManager,
    TrainingJobNotFoundError,
)
from src.config.settings import Settings
from src.storage.model_run_fact_repository import (
    ModelRunFactRepositoryError,
    ModelRunFactViewNotFoundError,
)


class _FakeModelRunRepository:
    def export_page(self, limit: int, offset: int):
        assert limit == 10
        assert offset == 2
        return 3, [{"run_id": "run-1", "test_pr_auc": 0.91}]


class _UnavailableModelRunRepository:
    def export_page(self, limit: int, offset: int):
        raise ModelRunFactRepositoryError("database unavailable")


class _MissingViewRepository:
    def export_page(self, limit: int, offset: int):
        raise ModelRunFactViewNotFoundError("view missing")


class _FakeTrainingJobManager:
    def __init__(self) -> None:
        self.request: TrainingRequest | None = None

    def start(self, request: TrainingRequest):
        self.request = request
        return self._job()

    def get(self, job_id: str):
        assert job_id == "job-1"
        return self._job()

    @staticmethod
    def _job():
        return {
            "job_id": "job-1",
            "status": "queued",
            "created_at": datetime(2026, 6, 14, tzinfo=timezone.utc),
            "started_at": None,
            "completed_at": None,
            "configuration": {
                "false_positive_cost": 2.0,
                "training_max_rows": None,
            },
            "result": None,
            "error": None,
        }


class _BusyTrainingJobManager:
    def start(self, request: TrainingRequest):
        raise TrainingAlreadyRunningError("job-active")


class _MissingTrainingJobManager:
    def get(self, job_id: str):
        raise TrainingJobNotFoundError(job_id)


def test_export_model_runs_returns_paginated_json() -> None:
    app.dependency_overrides[get_model_run_fact_repository] = _FakeModelRunRepository
    try:
        response = TestClient(app).get("/model-runs/export?limit=10&offset=2")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        'attachment; filename="fact_model_runs.json"'
    )
    assert response.json() == {
        "source": "fraud_tracking.fact_model_runs",
        "total": 3,
        "count": 1,
        "limit": 10,
        "offset": 2,
        "items": [{"run_id": "run-1", "test_pr_auc": 0.91}],
    }


def test_export_model_runs_returns_503_when_database_is_unavailable() -> None:
    app.dependency_overrides[get_model_run_fact_repository] = _UnavailableModelRunRepository
    try:
        response = TestClient(app).get("/model-runs/export")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "PostgreSQL indisponivel" in response.json()["detail"]


def test_export_model_runs_explains_missing_migration() -> None:
    app.dependency_overrides[get_model_run_fact_repository] = _MissingViewRepository
    try:
        response = TestClient(app).get("/model-runs/export")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "scripts.migrate_database upgrade" in response.json()["detail"]


def test_start_training_accepts_env_style_aliases_and_returns_202() -> None:
    manager = _FakeTrainingJobManager()
    app.dependency_overrides[get_training_job_manager] = lambda: manager
    try:
        response = TestClient(app).post(
            "/training-runs",
            json={
                "FALSE_POSITIVE_COST": 2,
                "THRESHOLD_COST_SCENARIOS": "1:10,5:25",
                "TRAINING_MAX_ROWS": 0,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-1"
    assert manager.request is not None
    assert manager.request.threshold_cost_scenarios == ((1.0, 10.0), (5.0, 25.0))
    assert manager.request.settings_overrides()["training_max_rows"] is None


def test_start_training_allows_empty_body_to_use_env_defaults() -> None:
    manager = _FakeTrainingJobManager()
    app.dependency_overrides[get_training_job_manager] = lambda: manager
    try:
        response = TestClient(app).post("/training-runs")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert manager.request is not None
    assert manager.request.settings_overrides() == {}


def test_start_training_returns_conflict_for_concurrent_job() -> None:
    app.dependency_overrides[get_training_job_manager] = _BusyTrainingJobManager
    try:
        response = TestClient(app).post("/training-runs", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert "job-active" in response.json()["detail"]


def test_get_training_status_returns_job_snapshot() -> None:
    app.dependency_overrides[get_training_job_manager] = _FakeTrainingJobManager
    try:
        response = TestClient(app).get("/training-runs/job-1")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_get_training_status_returns_404_for_unknown_job() -> None:
    app.dependency_overrides[get_training_job_manager] = _MissingTrainingJobManager
    try:
        response = TestClient(app).get("/training-runs/unknown")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_training_request_keeps_settings_defaults_for_omitted_fields() -> None:
    defaults = Settings(
        false_negative_cost=37.0,
        promotion_min_recall=0.88,
    )
    request = TrainingRequest(FALSE_POSITIVE_COST=3)
    resolved = {
        **{
            "false_negative_cost": defaults.false_negative_cost,
            "promotion_min_recall": defaults.promotion_min_recall,
        },
        **request.settings_overrides(),
    }

    assert resolved["false_positive_cost"] == 3
    assert resolved["false_negative_cost"] == 37
    assert resolved["promotion_min_recall"] == 0.88


def test_training_job_manager_resolves_defaults_and_uses_isolated_staging(
    tmp_path,
    monkeypatch,
) -> None:
    import src.api.training_service as training_service

    base_settings = Settings(
        project_root=tmp_path,
        false_negative_cost=37,
        training_max_rows=123,
    )
    captured = {}
    completed = []

    class _FakeTrainingPipeline:
        def __init__(self, settings):
            captured["settings"] = settings

        def run(self):
            return SimpleNamespace(
                run_id="run-1",
                model_name="logistic_regression",
                threshold=0.2,
                baseline_decision="reject",
                validation_metrics={"pr_auc": 0.5},
                test_metrics={"pr_auc": 0.4},
                out_of_time_metrics={"pr_auc": 0.3},
            )

    monkeypatch.setattr(training_service, "Settings", lambda: base_settings)
    monkeypatch.setattr(training_service, "TrainingPipeline", _FakeTrainingPipeline)
    manager = TrainingJobManager(on_complete=lambda: completed.append(True))
    try:
        job = manager.start(TrainingRequest(FALSE_POSITIVE_COST=3))
        for _ in range(100):
            job = manager.get(job["job_id"])
            if job["status"] in {"completed", "failed"}:
                break
            sleep(0.01)
    finally:
        manager.close()

    assert job["status"] == "completed"
    assert job["configuration"]["false_positive_cost"] == 3
    assert job["configuration"]["false_negative_cost"] == 37
    assert job["configuration"]["training_max_rows"] == 123
    assert captured["settings"].artifacts_dir.parent.name == job["job_id"]
    assert completed == [True]
