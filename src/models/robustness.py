"""Optional geographic feature ablation experiments."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.feature_report import build_feature_importance
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer


GEO_CORE = ("merchant_city", "merchant_state")
GEO_ALL = GEO_CORE + ("zip", "latitude", "longitude")


def run_geographic_ablation(
    settings: Settings,
    parent_run_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    primary_metrics: dict,
    primary_top_features: pd.DataFrame,
) -> pd.DataFrame:
    """Train controlled variants to quantify reliance on geographic features."""
    rows = [
        _result_row(
            parent_run_id,
            "A_full",
            (),
            settings.feature_set_version,
            primary_metrics,
            primary_top_features,
        )
    ]
    experiments = (
        ("B_without_city_state", GEO_CORE, settings.categorical_min_frequency),
        ("C_without_all_geo", GEO_ALL, settings.categorical_min_frequency),
        ("D_without_geo_rare_categories", GEO_ALL, 100),
    )
    thresholds = threshold_grid(
        settings.threshold_analysis_start,
        settings.threshold_analysis_stop,
        settings.threshold_analysis_step,
    )
    for experiment_group, exclusions, min_frequency in experiments:
        experiment_settings = replace(
            settings,
            feature_exclusions=tuple(exclusions),
            categorical_min_frequency=min_frequency,
            run_geo_ablation=False,
            feature_set_version=f"{settings.feature_set_version}:{experiment_group}",
        )
        pipeline = FraudModelTrainer(experiment_settings).train(X_train, y_train)
        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        validation_table = build_threshold_table(
            y_validation.to_numpy(),
            validation_scores,
            thresholds=thresholds,
            beta=settings.threshold_beta,
            false_positive_cost=settings.false_positive_cost,
            false_negative_cost=settings.false_negative_cost,
            split="validation",
        )
        threshold, _ = select_business_threshold(validation_table)
        test_scores = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary_classifier(
            y_test.to_numpy(),
            test_scores,
            threshold=threshold,
            beta=settings.threshold_beta,
        )
        metrics["business_cost"] = (
            metrics["fp"] * settings.false_positive_cost
            + metrics["fn"] * settings.false_negative_cost
        )
        rows.append(
            _result_row(
                parent_run_id,
                experiment_group,
                exclusions,
                experiment_settings.feature_set_version,
                metrics,
                build_feature_importance(pipeline),
            )
        )
    return pd.DataFrame(rows)


def _result_row(
    parent_run_id: str,
    experiment_group: str,
    features_removed: tuple[str, ...],
    feature_set_version: str,
    metrics: dict,
    top_features: pd.DataFrame,
) -> dict:
    return {
        "parent_run_id": parent_run_id,
        "experiment_run_id": f"{parent_run_id}:{experiment_group}",
        "experiment_group": experiment_group,
        "feature_set_version": feature_set_version,
        "features_removed": json.dumps(list(features_removed)),
        "threshold": metrics["threshold"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fbeta": metrics["fbeta"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "business_cost": metrics.get("business_cost"),
        "alert_rate": metrics["alert_rate"],
        "top_features": top_features.head(20).to_json(orient="records"),
    }
