"""Tests for governed review reports, target audit and drift."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data.split_data import DataSplits
from src.models.baseline_decision import BaselineDecisionService
from src.models.data_drift import DataDriftReportService
from src.models.robustness import geographic_feature_names, write_robustness_reports
from src.models.target_audit import TargetAuditService
from src.models.threshold_analysis import build_threshold_recommendations, build_threshold_table


def _splits() -> DataSplits:
    frame = pd.DataFrame(
        {
            "transaction_id": [str(i) for i in range(12)],
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "amount": [10, 11, 12, 13, 100, 120, 14, 15, 16, 17, 200, 220],
            "merchant_state": ["SP", "SP", "RJ", "RJ", "AM", "AM"] * 2,
            "zip": [1000, 1001, 2000, 2001, 3000, 3001] * 2,
            "is_fraud": [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        }
    )
    return DataSplits(
        train=frame.iloc[:5].copy(),
        validation=frame.iloc[5:8].copy(),
        test=frame.iloc[8:10].copy(),
        out_of_time=frame.iloc[10:].copy(),
        time_column="date",
    )


def test_geographic_feature_detection() -> None:
    assert geographic_feature_names(
        ("amount", "merchant_city", "merchant_state", "zip", "latitude", "mcc")
    ) == ("merchant_city", "merchant_state", "zip", "latitude")


def test_target_audit_flags_missing_labels(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    transactions = pd.DataFrame(
        {
            "id": [str(i) for i in range(14)],
            "date": pd.date_range("2020-01-01", periods=14, freq="D"),
        }
    )
    labels = {"target": {str(i): ("Yes" if i in {2, 8} else "No") for i in range(12)}}

    payload = TargetAuditService(settings).build(
        transactions,
        labels,
        _splits(),
        tmp_path,
    )

    assert payload["status"] == "warning"
    assert payload["missing_label_count"] == 2
    assert payload["unknown_as_negative_risk"] is True
    assert (tmp_path / settings.target_audit_filename).exists()
    assert (tmp_path / settings.target_audit_by_split_filename).exists()


def test_data_drift_report_writes_json_markdown_and_csv(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)

    payload = DataDriftReportService(settings).build(_splits(), tmp_path)

    assert payload["numeric_feature_count"] >= 1
    assert payload["categorical_feature_count"] >= 1
    assert (tmp_path / settings.data_drift_report_filename).exists()
    assert (tmp_path / settings.data_drift_numeric_filename).exists()
    assert (tmp_path / settings.data_drift_categorical_filename).exists()


def test_threshold_recommendations_include_validation_and_retrospective_oot() -> None:
    thresholds = np.array([0.2, 0.5])
    table = pd.concat(
        [
            build_threshold_table(
                np.array([0, 1, 1]),
                np.array([0.1, 0.4, 0.9]),
                thresholds,
                beta=2,
                false_positive_cost=1,
                false_negative_cost=25,
                split=split,
            )
            for split in ("validation", "test", "out_of_time")
        ],
        ignore_index=True,
    )

    recommendations = build_threshold_recommendations(table, max_alert_rate=1.0)

    assert recommendations["validation_lowest_cost"] is not None
    assert recommendations["most_stable"] is not None
    assert recommendations["retrospective_out_of_time_lowest_cost"] is not None


def test_baseline_decision_rejects_bad_out_of_time_and_pending_review_for_warnings(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, promotion_min_recall=0.5)
    metadata = {
        "validation_metrics": {"pr_auc": 0.5, "recall": 0.8},
        "test_metrics": {"pr_auc": 0.4, "recall": 0.7, "tp": 1, "fp": 1, "tn": 1, "fn": 1},
        "out_of_time_metrics": {"pr_auc": 0.1, "recall": 0.1, "alert_rate": 0.01},
    }
    leakage = {"status": "pass", "checks": {}, "warnings": []}

    rejected = BaselineDecisionService(settings).decide(metadata, leakage, [])

    assert rejected["decision"] == "reject"
    assert rejected["blocking_reasons"]

    better_metadata = {
        **metadata,
        "out_of_time_metrics": {"pr_auc": 0.49, "recall": 0.7, "alert_rate": 0.01},
    }
    pending = BaselineDecisionService(settings).decide(
        better_metadata,
        {"status": "pass", "checks": {}, "warnings": ["review geo"]},
        [],
    )

    assert pending["decision"] == "pending_review"
    assert pending["warnings"]


def test_disabled_robustness_report_is_written(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)

    payload = write_robustness_reports(pd.DataFrame(), tmp_path, settings)

    assert payload["status"] == "disabled"
    assert json.loads((tmp_path / settings.robustness_report_filename).read_text())["status"] == "disabled"
    assert (tmp_path / settings.geo_ablation_report_filename).exists()
