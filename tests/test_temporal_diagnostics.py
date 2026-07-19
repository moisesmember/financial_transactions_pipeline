"""Tests for period performance and operational Top-K diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.temporal_performance import write_temporal_performance_artifacts
from src.models.top_k_analysis import write_top_k_analysis


def test_performance_by_year_and_month_artifacts_are_generated(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2018-01-01", "2018-02-01", "2019-01-01", "2019-02-01"]
            )
        }
    )
    targets = pd.Series([0, 1, 1, 0])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    year_path = tmp_path / "performance_by_year.csv"
    month_path = tmp_path / "performance_by_month.csv"
    markdown_path = tmp_path / "performance_by_period.md"

    result = write_temporal_performance_artifacts(
        {"test": frame, "out_of_time": frame},
        {"test": targets, "out_of_time": targets},
        {"test": scores, "out_of_time": scores},
        "date",
        threshold=0.5,
        beta=2.0,
        false_positive_cost=1.0,
        false_negative_cost=25.0,
        year_path=year_path,
        month_path=month_path,
        markdown_path=markdown_path,
    )

    required = {
        "rows",
        "positive_count",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "pr_auc",
        "roc_auc",
        "business_cost",
        "mean_score",
        "median_score",
        "threshold",
    }
    assert required <= set(result["year"].columns)
    assert set(result["year"]["period"]) == {"2018", "2019"}
    assert year_path.exists() and month_path.exists() and markdown_path.exists()


def test_top_k_analysis_obeys_alert_capacity_and_never_changes_decision(tmp_path) -> None:
    target = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

    result = write_top_k_analysis(
        {"out_of_time": (target, scores)},
        (1, 3, 10),
        false_positive_cost=1.0,
        false_negative_cost=25.0,
        csv_path=tmp_path / "top_k_analysis.csv",
        markdown_path=tmp_path / "top_k_analysis.md",
    )

    top_one = result.loc[result["requested_k"].eq(1)].iloc[0]
    assert top_one["frauds_found"] == 1
    assert top_one["recall_at_k"] == 0.5
    assert result.loc[result["requested_k"].eq(10), "effective_k"].iloc[0] == 5
    assert "decision" not in result.columns
