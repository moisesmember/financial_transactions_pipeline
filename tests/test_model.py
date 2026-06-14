"""Tests for model factory, thresholding and training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.model_factory import ModelFactory
from src.models.threshold import find_best_threshold
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer


def test_model_factory_creates_supported_model() -> None:
    """Factory should create a configured classifier."""
    model = ModelFactory(Settings()).create("logistic_regression")

    assert hasattr(model, "fit")


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
