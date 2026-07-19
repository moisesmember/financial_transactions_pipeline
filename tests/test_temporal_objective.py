"""Regression tests for eligibility-first temporal model selection."""

from __future__ import annotations

import json

import numpy as np

from src.models.baseline_challenger import compare_baseline_challenger
from src.models.evaluate import evaluate_binary_classifier
from src.models.temporal_objective import (
    INELIGIBLE_OBJECTIVE_SCORE,
    temporal_robustness_score,
    write_objective_breakdown,
)
from src.models.threshold_analysis import threshold_at_search_boundary


def _fold(pr_auc: float, prevalence: float, recall: float, precision: float = 0.2):
    return {
        "pr_auc": pr_auc,
        "fraud_rate": prevalence,
        "row_count": 1000,
        "metrics": {
            "evaluation_status": "valid",
            "recall": recall,
            "precision": precision,
            "alert_rate": 0.02,
            "business_cost": 100.0,
            "cost_per_record": 0.1,
        },
    }


def test_no_positive_period_has_null_recall_pr_auc_and_fbeta() -> None:
    metrics = evaluate_binary_classifier(
        np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.7, 0.9]), threshold=0.5
    )

    assert metrics["evaluation_status"] == "no_positive_class"
    assert metrics["recall"] is None
    assert metrics["pr_auc"] is None
    assert metrics["fbeta"] is None
    assert metrics["fp"] == 2


def test_only_positive_period_does_not_report_ranking_auc() -> None:
    metrics = evaluate_binary_classifier(
        np.ones(3, dtype=int), np.array([0.2, 0.7, 0.9]), threshold=0.5
    )

    assert metrics["evaluation_status"] == "no_negative_class"
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["recall"] is not None


def test_invalid_periods_do_not_enter_temporal_averages() -> None:
    invalid = {
        "pr_auc": None,
        "fraud_rate": 0.0,
        "metrics": {
            "evaluation_status": "no_positive_class",
            "recall": None,
            "precision": 0.0,
            "alert_rate": 0.01,
        },
    }
    score = temporal_robustness_score(
        [_fold(0.2, 0.01, 0.6), invalid, _fold(0.1, 0.01, 0.4)],
        min_valid_fold_count=2,
        max_alert_rate=1.0,
    )

    assert score["valid_fold_count"] == 2
    assert score["single_class_fold_count"] == 1
    assert score["no_positive_period_count"] == 1
    assert score["mean_fold_recall"] == 0.5


def test_ineligible_trial_receives_floor_and_cannot_beat_eligible_trial() -> None:
    eligible = temporal_robustness_score(
        [_fold(0.20, 0.01, 0.7), _fold(0.18, 0.01, 0.65), _fold(0.17, 0.01, 0.6)]
    )
    ineligible = temporal_robustness_score(
        [_fold(0.20, 0.01, 0.9), _fold(0.01, 0.01, 0.01), _fold(0.01, 0.01, 0.01)]
    )

    assert eligible["eligibility_status"] == "eligible"
    assert ineligible["eligibility_status"] == "ineligible"
    assert ineligible["final_objective_score"] == INELIGIBLE_OBJECTIVE_SCORE
    assert eligible["final_objective_score"] > ineligible["final_objective_score"]


def test_temporal_instability_is_subtracted_and_score_is_reproducible(tmp_path) -> None:
    stable_folds = [_fold(0.20, 0.01, 0.6), _fold(0.19, 0.01, 0.6), _fold(0.18, 0.01, 0.6)]
    unstable_folds = [_fold(0.30, 0.01, 0.8), _fold(0.08, 0.01, 0.3), _fold(0.18, 0.01, 0.6)]
    stable = temporal_robustness_score(stable_folds, max_pr_auc_temporal_drop=1.0, max_recall_temporal_drop=1.0)
    unstable = temporal_robustness_score(unstable_folds, max_pr_auc_temporal_drop=1.0, max_recall_temporal_drop=1.0)

    assert unstable["temporal_instability_penalty"] > stable["temporal_instability_penalty"]
    assert stable == temporal_robustness_score(stable_folds, max_pr_auc_temporal_drop=1.0, max_recall_temporal_drop=1.0)

    row = {"trial_number": 0, "model_name": "model", **stable}
    write_objective_breakdown([row], tmp_path / "breakdown.csv", tmp_path / "breakdown.md")
    assert (tmp_path / "breakdown.csv").exists()
    assert "Formula" in (tmp_path / "breakdown.md").read_text(encoding="utf-8")


def test_worse_challenger_never_replaces_rejected_baseline() -> None:
    baseline = {
        "model_name": "hist_gradient_boosting",
        "dataset": {f"{split}_positive_rate": 0.01 for split in ("validation", "test", "out_of_time")},
    }
    challenger = {
        "model_name": "logistic_regression_regularized",
        "dataset": dict(baseline["dataset"]),
    }
    for split in ("validation", "test", "out_of_time"):
        baseline[f"{split}_metrics"] = {
            "pr_auc": 0.02, "precision": 0.02, "recall": 0.06,
            "tp": 60, "fp": 3000, "fn": 940, "alert_rate": 0.01,
        }
        challenger[f"{split}_metrics"] = {
            "pr_auc": 0.003, "precision": 0.004, "recall": 0.04,
            "tp": 40, "fp": 9000, "fn": 960, "alert_rate": 0.03,
        }

    result = compare_baseline_challenger(baseline, challenger)

    assert result["challenger_replaces_baseline"] is False
    assert result["decision"] == "keep_best_rejected_baseline"
    assert result["best_rejected_baseline"]["role"] == "best_rejected_baseline"


def test_threshold_boundary_is_explicitly_detected() -> None:
    assert threshold_at_search_boundary(0.99, 0.01, 0.99) is True
    assert threshold_at_search_boundary(0.50, 0.01, 0.99) is False
