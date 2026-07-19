"""Tests for governed review reports, target audit and drift."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data.split_data import DataSplits
from src.models.baseline_decision import BaselineDecisionService
from src.models.data_drift import DataDriftReportService
from src.models.error_attribution import build_error_attribution_report
from src.models.robustness import (
    REQUIRED_GEO_EXPERIMENTS,
    geographic_feature_names,
    write_robustness_reports,
)
from src.models.target_audit import TargetAuditService
from src.models.threshold_analysis import build_threshold_recommendations, build_threshold_table
from src.models.walk_forward import summarize_walk_forward_folds


def test_error_attribution_report_describes_all_confusion_cohorts(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "amount": [10.0, 20.0, 100.0, 200.0],
            "use_chip": ["Chip", "Swipe", "Online", "Online"],
            "mcc": [1, 2, 3, 4],
            "merchant_city": ["A", "B", "C", "D"],
            "merchant_state": ["AA", "BB", "CC", "DD"],
            "transaction_hour": [1, 2, 3, 4],
        }
    )
    output = tmp_path / "error_attribution_report.json"
    markdown = tmp_path / "error_attribution_report.md"
    groups = tmp_path / "error_attribution_by_group.csv"

    report = build_error_attribution_report(
        {"out_of_time": frame},
        {"out_of_time": pd.Series([1, 0, 1, 0])},
        {"out_of_time": np.array([0.9, 0.8, 0.2, 0.1])},
        threshold=0.5,
        output_path=output,
        markdown_path=markdown,
        by_group_path=groups,
    )

    cohorts = report["splits"]["out_of_time"]
    assert [cohorts[name]["count"] for name in ("TP", "FP", "FN", "TN")] == [1, 1, 1, 1]
    assert cohorts["FP"]["amount_mean"] == 20.0
    assert cohorts["FN"]["amount_mean"] == 100.0
    assert cohorts["fp_vs_fn"]["amount_mean_difference_fp_minus_fn"] == -80.0
    assert report["shap_status"] == "not_computed"
    assert output.exists()
    assert markdown.exists()
    assert groups.exists()
    assert cohorts["FN"]["score_percentiles"]


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
    assert set(REQUIRED_GEO_EXPERIMENTS) == {
        "A_full",
        "B_without_coordinates",
        "C_without_city_state",
        "D_without_all_geo",
        "E_transactional_behavioral_only",
        "F_city_grouped_online_offline_other",
        "G_without_high_drift_features",
    }


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
    assert payload["target_status"] == "valid_with_coverage_warning"
    assert payload["target_valid"] is True
    assert payload["coverage_warning"] is True
    assert payload["missing_label_count"] == 2
    assert payload["missing_transaction_labels_count"] == 2
    assert payload["missing_transaction_labels_removed"] is True
    assert payload["supervised_join_type"] == "inner"
    assert payload["label_join_policy"] == "inner_join"
    assert payload["unlabeled_transaction_policy"] == "removed_by_inner_join"
    assert payload["unlabeled_as_negative"] is False
    assert payload["unknown_labels_used_as_negative"] is False
    assert payload["invalid_label_values_found"] is False
    assert payload["invalid_label_policy"] == "raise_error"
    assert payload["unknown_as_negative_risk"] is False
    assert any("O target e valido, mas ha warning de cobertura" in item for item in payload["warnings"])
    markdown = (tmp_path / settings.target_audit_markdown_filename).read_text(encoding="utf-8")
    assert "Transacoes sem label foram removidas via inner join." in markdown
    assert "Nenhuma transacao sem label foi convertida para classe 0." in markdown
    assert "Labels invalidos geram erro antes do treino." in markdown
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
    assert (tmp_path / settings.feature_stability_report_filename).exists()
    assert (tmp_path / settings.feature_stability_markdown_filename).exists()
    assert (tmp_path / settings.feature_stability_by_period_filename).exists()


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


def test_walk_forward_summary_flags_bad_last_fold() -> None:
    summary = summarize_walk_forward_folds(
        [
            {"fold": 1, "recall": 0.70, "pr_auc": 0.40, "fraud_rate": 0.01},
            {"fold": 2, "recall": 0.60, "pr_auc": 0.35, "fraud_rate": 0.01},
            {"fold": 3, "recall": 0.01, "pr_auc": 0.005, "fraud_rate": 0.01},
        ],
        min_recall=0.05,
        min_pr_auc_lift=1.0,
        max_recall_drop=0.50,
    )

    assert summary["worst_fold"] == 3
    assert summary["last_fold_recall"] == 0.01
    assert summary["last_fold_penalty"] > 0
    assert summary["unstable"] is True
    assert summary["failure_reasons"]


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
