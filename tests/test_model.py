"""Tests for model factory, thresholding and training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.model_factory import ModelFactory
from src.models.external_benchmarks import ExternalBenchmarkRunner
from src.models.optuna_search import OptunaModelSelector
from src.models.threshold import find_best_threshold
from src.models.threshold_analysis import (
    build_cost_scenario_summary,
    build_threshold_table,
    select_business_threshold,
    threshold_grid,
)
from src.models.train import FraudModelTrainer


def test_model_factory_creates_supported_model() -> None:
    """Factory should create a configured classifier."""
    model = ModelFactory(Settings()).create("logistic_regression")

    assert hasattr(model, "fit")


def test_model_factory_applies_parameter_overrides() -> None:
    model = ModelFactory(Settings()).create(
        "logistic_regression",
        params={"C": 0.25},
    )

    assert model.C == 0.25


def test_settings_accept_optional_optuna_models() -> None:
    settings = Settings(
        optuna_model_candidates=("xgboost", "lightgbm", "catboost"),
    )

    assert set(settings.optuna_model_candidates) == {"xgboost", "lightgbm", "catboost"}


def test_model_factory_reports_unavailable_optional_models(monkeypatch) -> None:
    strategy = ModelFactory._strategies["catboost"]
    monkeypatch.setattr(strategy, "is_available", lambda: False)

    assert ModelFactory.unavailable_model_names(("logistic_regression", "catboost")) == (
        "catboost",
    )


def test_threshold_optimizes_fbeta() -> None:
    """Threshold tuning should return a valid threshold and metrics."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.05, 0.2, 0.7, 0.9])

    threshold, metrics = find_best_threshold(y_true, y_score, beta=2.0)

    assert 0.0 <= threshold <= 1.0
    assert metrics["fbeta"] > 0.0


def test_evaluate_includes_pr_auc_not_accuracy() -> None:
    """Evaluation should expose fraud-oriented metrics and omit accuracy."""
    metrics = evaluate_binary_classifier(np.array([0, 1]), np.array([0.1, 0.9]), threshold=0.5)

    assert "pr_auc" in metrics
    assert "recall" in metrics
    assert "accuracy" not in metrics


def test_business_threshold_table_compares_confusion_costs() -> None:
    """Business threshold selection should minimize configured validation cost."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.05, 0.15, 0.25, 0.20, 0.90])
    table = build_threshold_table(
        y_true,
        y_score,
        thresholds=np.array([0.10, 0.30]),
        beta=2.0,
        false_positive_cost=1.0,
        false_negative_cost=10.0,
        split="validation",
    )

    threshold, metrics = select_business_threshold(table)

    assert threshold == 0.10
    assert {"tp", "fp", "fn", "business_cost"}.issubset(table.columns)
    assert metrics["business_cost"] == 2.0


def test_threshold_grid_includes_requested_endpoints() -> None:
    grid = threshold_grid(0.08, 0.30, 0.01)

    assert grid[0] == 0.08
    assert grid[-1] == 0.30


def test_cost_scenarios_select_threshold_on_validation_for_every_split() -> None:
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.20, 0.60, 0.90])

    summary = build_cost_scenario_summary(
        {
            "validation": (y_true, scores),
            "test": (y_true, scores),
            "out_of_time": (y_true, scores),
        },
        thresholds=np.array([0.10, 0.50]),
        beta=2.0,
        cost_scenarios=((1.0, 10.0), (5.0, 25.0)),
    )

    assert len(summary) == 6
    assert set(summary["split"]) == {"validation", "test", "out_of_time"}
    assert set(summary["scenario_name"]) == {"fp_1_fn_10", "fp_5_fn_25"}


def test_training_pipeline_can_fit_small_dataframe() -> None:
    """Trainer should fit the complete sklearn pipeline on tabular data."""
    settings = Settings(model_name="logistic_regression")
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="D"),
            "transaction_id": [str(i) for i in range(12)],
            "card_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "merchant_id": [100, 101, 100, 102, 100, 103, 103, 104, 100, 105, 105, 106],
            "amount": [5, 7, 100, 6, 8, 120, 9, 11, 4, 130, 10, 12],
            "merchant_state": ["SP", "SP", "RJ", "SP", "RJ", "RJ", "SP", "SP", "SP", "RJ", "SP", "RJ"],
        }
    )
    y = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])

    pipeline = FraudModelTrainer(settings).train(frame, y)
    scores = pipeline.predict_proba(frame)[:, 1]

    assert len(scores) == len(frame)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_hist_gradient_boosting_uses_dense_compatible_preprocessing() -> None:
    settings = Settings(model_name="hist_gradient_boosting")
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20, freq="D"),
            "card_id": [index % 4 for index in range(20)],
            "amount": [float(index) for index in range(20)],
            "merchant_state": ["SP", "RJ"] * 10,
        }
    )
    target = pd.Series([0, 1] * 10)

    pipeline = FraudModelTrainer(settings).train(frame, target)

    assert len(pipeline.predict_proba(frame)) == len(frame)


def test_external_benchmarks_record_unavailable_dependencies(tmp_path, monkeypatch) -> None:
    settings = Settings(
        project_root=tmp_path,
        external_benchmark_backends=("autogluon", "h2o", "flaml"),
    )
    runner = ExternalBenchmarkRunner(settings)
    for adapter_type in runner._adapters.values():
        monkeypatch.setattr(adapter_type, "is_available", lambda self: False)

    summary = runner.run(
        dataset=None,
        results_path=tmp_path / "results.csv",
        summary_path=tmp_path / "summary.json",
        output_dir=tmp_path / "output",
    )

    assert {item["backend"] for item in summary} == {"autogluon", "h2o", "flaml"}
    assert {item["status"] for item in summary} == {"unavailable"}


def test_optuna_search_executes_each_supported_model_family(tmp_path) -> None:
    row_count = 120
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=row_count, freq="h"),
            "card_id": [index % 8 for index in range(row_count)],
            "amount": [100.0 if index % 10 == 0 else 10.0 + index % 5 for index in range(row_count)],
            "merchant_state": ["SP", "RJ", "AM"] * 40,
        }
    )
    target = pd.Series([1 if index % 10 == 0 else 0 for index in range(row_count)])
    settings = Settings(
        project_root=tmp_path,
        optuna_model_candidates=(
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
        ),
        optuna_trials=3,
        optuna_timeout_seconds=60,
        threshold_analysis_start=0.10,
        threshold_analysis_stop=0.90,
        threshold_analysis_step=0.20,
        categorical_min_frequency=2,
    )

    result = OptunaModelSelector(settings).select(
        frame.iloc[:80],
        target.iloc[:80],
        frame.iloc[80:],
        target.iloc[80:],
        trials_path=tmp_path / "trials.csv",
        study_path=tmp_path / "study.json",
    )
    trials = pd.read_csv(tmp_path / "trials.csv")

    assert set(trials["model_name"]) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert set(trials["state"]) == {"COMPLETE"}
    assert result.model_name in set(trials["model_name"])
