"""Geographic ablation and robustness experiments."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.feature_report import build_feature_importance
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer


GEO_CORE = ("merchant_city", "merchant_state")
GEO_COORDINATES = ("zip", "address", "latitude", "longitude")
GEO_ALL = GEO_CORE + GEO_COORDINATES
TRANSACTIONAL_BEHAVIORAL_KEEP_TOKENS = (
    "amount",
    "mcc",
    "chip",
    "hour",
    "day",
    "month",
    "weekend",
    "previous",
    "mean",
    "std",
    "transactions_seen",
)
GEO_TOKEN_HINTS = (
    "zip",
    "latitude",
    "longitude",
    "merchant_city",
    "merchant_state",
    "city",
    "state",
    "country",
    "address",
    "location",
)


def geographic_feature_names(columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return columns that encode geographic or location information."""
    return tuple(
        column
        for column in columns
        if any(token in column.lower() for token in GEO_TOKEN_HINTS)
    )


def run_geographic_ablation(
    settings: Settings,
    parent_run_id: str,
    model_name: str,
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_out_of_time: pd.DataFrame,
    y_out_of_time: pd.Series,
    primary_metrics_by_split: dict[str, dict[str, float]],
    primary_top_features: pd.DataFrame,
) -> pd.DataFrame:
    """Train controlled variants to quantify geographic overfitting."""
    started = datetime.now(timezone.utc)
    rows = _rows_for_existing_model(
        parent_run_id,
        "A_full",
        (),
        settings.feature_set_version,
        primary_metrics_by_split,
        primary_top_features,
        started,
    )
    available_geo = geographic_feature_names(tuple(X_train.columns))
    transactional_exclusions = tuple(
        column
        for column in X_train.columns
        if column != settings.target_column
        and not any(token in column.lower() for token in TRANSACTIONAL_BEHAVIORAL_KEEP_TOKENS)
    )
    experiments = (
        ("B_without_coordinates", tuple(col for col in GEO_COORDINATES if col in X_train.columns), model_name, model_params),
        ("C_without_city_state", tuple(col for col in GEO_CORE if col in X_train.columns), model_name, model_params),
        ("D_without_all_geo", available_geo, model_name, model_params),
        ("E_transactional_behavioral_only", transactional_exclusions, model_name, model_params),
        ("F_interpretable_baseline", available_geo, "logistic_regression", settings.model_params["logistic_regression"]),
        (
            "G_without_transactions_seen_before",
            tuple(col for col in ("transactions_seen_before",) if col in X_train.columns),
            model_name,
            model_params,
        ),
        (
            "H_without_use_chip",
            tuple(col for col in ("use_chip",) if col in X_train.columns),
            model_name,
            model_params,
        ),
        (
            "I_without_city_state_and_transactions_seen",
            tuple(
                col
                for col in (*GEO_CORE, "transactions_seen_before")
                if col in X_train.columns
            ),
            model_name,
            model_params,
        ),
    )
    thresholds = threshold_grid(
        settings.threshold_analysis_start,
        settings.threshold_analysis_stop,
        settings.threshold_analysis_step,
    )
    for experiment_group, exclusions, experiment_model, experiment_params in experiments:
        if experiment_group != "E_transactional_behavioral_only" and not exclusions:
            rows.extend(
                _status_rows(
                    parent_run_id,
                    experiment_group,
                    exclusions,
                    settings.feature_set_version,
                    "skipped",
                    "Nenhuma feature aplicavel encontrada para remocao.",
                )
            )
            continue
        experiment_started = datetime.now(timezone.utc)
        try:
            experiment_settings = replace(
                settings,
                feature_exclusions=tuple(sorted(set(settings.feature_exclusions) | set(exclusions))),
                run_geo_ablation=False,
                model_name=experiment_model,
                feature_set_version=f"{settings.feature_set_version}:{experiment_group}",
            )
            pipeline = FraudModelTrainer(experiment_settings).train(
                X_train,
                y_train,
                model_name=experiment_model,
                model_params=experiment_params,
            )
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
            scores_by_split = {
                "train": (y_train, pipeline.predict_proba(X_train)[:, 1]),
                "validation": (y_validation, validation_scores),
                "test": (y_test, pipeline.predict_proba(X_test)[:, 1]),
                "out_of_time": (y_out_of_time, pipeline.predict_proba(X_out_of_time)[:, 1]),
            }
            top_features = build_feature_importance(pipeline)
            rows.extend(
                _result_rows(
                    parent_run_id,
                    experiment_group,
                    exclusions,
                    experiment_settings.feature_set_version,
                    experiment_model,
                    threshold,
                    scores_by_split,
                    top_features,
                    "completed",
                    None,
                    (datetime.now(timezone.utc) - experiment_started).total_seconds(),
                    settings,
                )
            )
        except Exception as exc:  # noqa: BLE001 - robustness should not fail main training
            rows.extend(
                _status_rows(
                    parent_run_id,
                    experiment_group,
                    exclusions,
                    settings.feature_set_version,
                    "failed",
                    str(exc),
                    (datetime.now(timezone.utc) - experiment_started).total_seconds(),
                )
            )
    return pd.DataFrame(rows)


def write_robustness_reports(
    results: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> dict[str, Any]:
    """Write robustness and geo ablation JSON/Markdown reports."""
    payload = _summarize_results(results)
    safe_payload = _json_safe(payload)
    robustness_json = output_dir / settings.robustness_report_filename
    robustness_md = output_dir / settings.robustness_markdown_filename
    geo_json = output_dir / settings.geo_ablation_report_filename
    geo_md = output_dir / settings.geo_ablation_markdown_filename
    robustness_json.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    robustness_md.write_text(_markdown("Robustness Report", safe_payload), encoding="utf-8")
    geo_json.write_text(
        json.dumps(safe_payload["geo_ablation"], indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    geo_md.write_text(_markdown("Geo Ablation Report", safe_payload["geo_ablation"]), encoding="utf-8")
    return safe_payload


def _rows_for_existing_model(
    parent_run_id: str,
    experiment_group: str,
    features_removed: tuple[str, ...],
    feature_set_version: str,
    metrics_by_split: dict[str, dict[str, float]],
    top_features: pd.DataFrame,
    started: datetime,
) -> list[dict[str, Any]]:
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    rows = []
    for split, metrics in metrics_by_split.items():
        rows.append(
            _result_row(
                parent_run_id,
                experiment_group,
                features_removed,
                feature_set_version,
                "selected_model",
                split,
                metrics,
                top_features,
                "completed",
                None,
                duration,
            )
        )
    return rows


def _result_rows(
    parent_run_id: str,
    experiment_group: str,
    features_removed: tuple[str, ...],
    feature_set_version: str,
    model_name: str,
    threshold: float,
    scores_by_split: dict[str, tuple[pd.Series, Any]],
    top_features: pd.DataFrame,
    status: str,
    message: str | None,
    duration_seconds: float,
    settings: Settings,
) -> list[dict[str, Any]]:
    rows = []
    for split, (target, scores) in scores_by_split.items():
        metrics = evaluate_binary_classifier(
            target.to_numpy(),
            scores,
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
                features_removed,
                feature_set_version,
                model_name,
                split,
                metrics,
                top_features,
                status,
                message,
                duration_seconds,
            )
        )
    return rows


def _result_row(
    parent_run_id: str,
    experiment_group: str,
    features_removed: tuple[str, ...],
    feature_set_version: str,
    model_name: str,
    split: str,
    metrics: dict,
    top_features: pd.DataFrame,
    status: str,
    message: str | None,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "parent_run_id": parent_run_id,
        "experiment_run_id": f"{parent_run_id}:{experiment_group}",
        "experiment_group": experiment_group,
        "model_name": model_name,
        "split": split,
        "feature_set_version": feature_set_version,
        "features_removed": json.dumps(list(features_removed)),
        "feature_count_removed": len(features_removed),
        "threshold": metrics.get("threshold"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "fbeta": metrics.get("fbeta"),
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "tp": int(metrics.get("tp", 0)),
        "fp": int(metrics.get("fp", 0)),
        "tn": int(metrics.get("tn", 0)),
        "fn": int(metrics.get("fn", 0)),
        "business_cost": metrics.get("business_cost"),
        "alert_rate": metrics.get("alert_rate"),
        "status": status,
        "message": message,
        "duration_seconds": duration_seconds,
        "top_features": top_features.head(20).to_json(orient="records") if not top_features.empty else "[]",
    }


def _status_rows(
    parent_run_id: str,
    experiment_group: str,
    features_removed: tuple[str, ...],
    feature_set_version: str,
    status: str,
    message: str,
    duration_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    return [
        {
            "parent_run_id": parent_run_id,
            "experiment_run_id": f"{parent_run_id}:{experiment_group}",
            "experiment_group": experiment_group,
            "model_name": None,
            "split": split,
            "feature_set_version": feature_set_version,
            "features_removed": json.dumps(list(features_removed)),
            "feature_count_removed": len(features_removed),
            "status": status,
            "message": message,
            "duration_seconds": duration_seconds,
            "top_features": "[]",
        }
        for split in ("train", "validation", "test", "out_of_time")
    ]


def _summarize_results(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty or "status" not in results.columns:
        geo_payload = {
            "status": "disabled",
            "warnings": ["Experimentos de robustez nao executados nesta configuracao."],
            "recommended_experiment": None,
            "full_out_of_time": None,
            "without_geo_out_of_time": None,
            "experiment_count": 0,
            "completed_experiment_count": 0,
        }
        return {
            "status": "disabled",
            "warnings": geo_payload["warnings"],
            "recommended_experiment": None,
            "experiments": [],
            "geo_ablation": geo_payload,
        }
    completed_oot = results.loc[
        results["status"].eq("completed") & results["split"].eq("out_of_time")
    ].copy()
    if completed_oot.empty:
        recommended = None
    else:
        completed_oot["stability_score"] = completed_oot.apply(
            lambda row: _experiment_stability_score(results, row["experiment_group"]),
            axis=1,
        )
        recommended = completed_oot.sort_values(
            ["stability_score", "business_cost", "pr_auc"],
            ascending=[True, True, False],
        ).iloc[0].to_dict()
    full = _experiment_split(results, "A_full", "out_of_time")
    without_geo = _experiment_split(results, "D_without_all_geo", "out_of_time")
    warnings = []
    if full and without_geo:
        full_pr_auc = float(full.get("pr_auc") or 0)
        geo_pr_auc = float(without_geo.get("pr_auc") or 0)
        full_recall = float(full.get("recall") or 0)
        geo_recall = float(without_geo.get("recall") or 0)
        if geo_pr_auc >= full_pr_auc or geo_recall >= full_recall:
            warnings.append(
                "Remover features geograficas manteve ou melhorou desempenho OOT; revisar overfitting geografico."
            )
    geo_payload = {
        "status": "warning" if warnings else "pass",
        "warnings": warnings,
        "recommended_experiment": recommended,
        "full_out_of_time": full,
        "without_geo_out_of_time": without_geo,
        "experiment_count": int(results["experiment_group"].nunique()),
        "completed_experiment_count": int(results.loc[results["status"].eq("completed"), "experiment_group"].nunique()),
    }
    return {
        "status": geo_payload["status"],
        "warnings": warnings,
        "recommended_experiment": recommended,
        "experiments": results.to_dict(orient="records"),
        "geo_ablation": geo_payload,
    }


def _experiment_stability_score(results: pd.DataFrame, experiment_group: str) -> float:
    group = results.loc[
        results["experiment_group"].eq(experiment_group) & results["status"].eq("completed")
    ]
    if group.empty:
        return float("inf")
    return float(
        group["recall"].astype(float).std(ddof=0)
        + group["alert_rate"].astype(float).std(ddof=0)
        + group["business_cost"].astype(float).std(ddof=0) / max(1.0, group["business_cost"].astype(float).mean())
    )


def _experiment_split(results: pd.DataFrame, experiment_group: str, split: str) -> dict[str, Any] | None:
    rows = results.loc[
        results["experiment_group"].eq(experiment_group)
        & results["split"].eq(split)
        & results["status"].eq("completed")
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _markdown(title: str, payload: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Experimentos: {payload.get('experiment_count', 'n/a')}",
        f"- Experimentos completos: {payload.get('completed_experiment_count', 'n/a')}",
        "",
        "## Warnings",
        "",
        *[f"- {warning}" for warning in payload.get("warnings", [])],
    ]
    recommended = payload.get("recommended_experiment")
    if recommended:
        lines.extend(
            [
                "",
                "## Recommended Safe Candidate",
                "",
                f"- Experimento: `{recommended.get('experiment_group')}`",
                f"- Split: `{recommended.get('split')}`",
                f"- PR-AUC OOT: {recommended.get('pr_auc')}",
                f"- Recall OOT: {recommended.get('recall')}",
                f"- Custo OOT: {recommended.get('business_cost')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value
