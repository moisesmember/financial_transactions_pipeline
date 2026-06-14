"""Tests for PostgreSQL migration definitions."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = (
    PROJECT_ROOT
    / "migrations"
    / "versions"
    / "20260613_0001_create_fraud_tracking_schema.py"
)
FACT_VIEW_MIGRATION_PATH = (
    PROJECT_ROOT
    / "migrations"
    / "versions"
    / "20260614_0002_create_model_run_fact_view.py"
)
GOVERNANCE_MIGRATION_PATH = (
    PROJECT_ROOT
    / "migrations"
    / "versions"
    / "20260614_0003_expand_model_governance.py"
)
ROBUSTNESS_MIGRATION_PATH = (
    PROJECT_ROOT
    / "migrations"
    / "versions"
    / "20260614_0004_add_robustness_experiments.py"
)
MODEL_SEARCH_MIGRATION_PATH = (
    PROJECT_ROOT
    / "migrations"
    / "versions"
    / "20260614_0005_add_model_search_tracking.py"
)


def test_initial_migration_has_expected_revision_and_operations() -> None:
    """The initial revision should remain stable and reversible."""
    spec = importlib.util.spec_from_file_location("initial_tracking_migration", MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.revision == "20260613_0001"
    assert module.down_revision is None
    assert callable(module.upgrade)
    assert callable(module.downgrade)


def test_alembic_configuration_uses_dedicated_tracking_schema() -> None:
    env_content = (PROJECT_ROOT / "migrations" / "env.py").read_text(encoding="utf-8")
    migration_content = MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'TRACKING_SCHEMA = "fraud_tracking"' in env_content
    assert 'SCHEMA = "fraud_tracking"' in migration_content
    assert '"training_runs"' in migration_content
    assert '"threshold_evaluations"' in migration_content
    assert '"baseline_promotions"' in migration_content


def test_fact_view_migration_consolidates_run_information() -> None:
    content = FACT_VIEW_MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'revision: str = "20260614_0002"' in content
    assert 'down_revision: str | None = "20260613_0001"' in content
    assert 'VIEW = "fact_model_runs"' in content
    assert "threshold_evaluations" in content
    assert "baseline_promotions" in content
    assert "test_pr_auc_rank" in content


def test_governance_migration_adds_oot_audit_and_monitoring() -> None:
    content = GOVERNANCE_MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'revision: str = "20260614_0003"' in content
    assert 'down_revision: str | None = "20260614_0002"' in content
    assert '"out_of_time_rows"' in content
    assert '"leakage_audit_checks"' in content
    assert '"model_features"' in content
    assert '"model_predictions"' in content
    assert "out_of_time_pr_auc" in content


def test_robustness_migration_tracks_geo_ablation_experiments() -> None:
    content = ROBUSTNESS_MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'revision: str = "20260614_0004"' in content
    assert 'down_revision: str | None = "20260614_0003"' in content
    assert '"robustness_experiments"' in content


def test_model_search_migration_tracks_optuna_and_external_benchmarks() -> None:
    content = MODEL_SEARCH_MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'revision: str = "20260614_0005"' in content
    assert 'down_revision: str | None = "20260614_0004"' in content
    assert '"model_search_trials"' in content
    assert '"external_benchmark_results"' in content
