"""Tests for the FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app, get_model_run_fact_repository
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
