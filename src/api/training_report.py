"""Human-oriented assembly of persisted model training governance data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.features.preprocessing import (
    POST_EVENT_FEATURES,
    RAW_ID_HINTS,
    SENSITIVE_FEATURES,
    SNAPSHOT_FEATURES,
)


METRIC_NAMES = (
    "threshold",
    "precision",
    "recall",
    "f1",
    "fbeta",
    "pr_auc",
    "roc_auc",
    "tp",
    "fp",
    "tn",
    "fn",
    "alert_rate",
    "business_cost",
)


def build_training_report(data: dict[str, Any]) -> dict[str, Any]:
    """Convert database-oriented records into a transparent report."""
    run = data["run"]
    metadata = run.get("metadata") or {}
    audit = run.get("leakage_audit") or {}
    features = data["features"]
    metrics = {
        split: {
            metric: run.get(f"{split}_{metric}")
            for metric in METRIC_NAMES
        }
        for split in ("validation", "test", "out_of_time")
    }
    validation_pr_auc = metrics["validation"]["pr_auc"]
    test_pr_auc = metrics["test"]["pr_auc"]
    oot_pr_auc = metrics["out_of_time"]["pr_auc"]
    generalization = {
        "validation_to_test_pr_auc_drop": _difference(validation_pr_auc, test_pr_auc),
        "validation_to_out_of_time_pr_auc_drop": _difference(
            validation_pr_auc, oot_pr_auc
        ),
        "validation_to_test_relative_drop": _relative_drop(
            validation_pr_auc, test_pr_auc
        ),
        "validation_to_out_of_time_relative_drop": _relative_drop(
            validation_pr_auc, oot_pr_auc
        ),
        "warning": bool(
            validation_pr_auc is not None
            and (
                (test_pr_auc is not None and test_pr_auc < validation_pr_auc * 0.7)
                or (oot_pr_auc is not None and oot_pr_auc < validation_pr_auc * 0.7)
            )
        ),
    }
    selected_columns = audit.get("selected_input_columns", [])
    excluded_columns = audit.get("excluded_input_columns", [])
    risk_columns = audit.get("risk_columns", {})
    if not excluded_columns:
        excluded_columns = sorted(
            {
                column
                for columns in risk_columns.values()
                for column in (columns or [])
            }
        )
    model_names = {
        item["model_name"]
        for item in data["search_trials"]
        if item.get("model_name")
    }
    if run.get("model_name"):
        model_names.add(run["model_name"])
    top_feature = features[0]["feature_name"] if features else None
    summary = [
        f"Modelo selecionado: {run.get('model_name')}.",
        f"Decisao de baseline: {run.get('promotion_decision') or 'nao informada'}.",
        (
            "PR-AUC validacao/teste/OOT: "
            f"{_format_metric(validation_pr_auc)} / {_format_metric(test_pr_auc)} / "
            f"{_format_metric(oot_pr_auc)}."
        ),
        f"Status da auditoria: {run.get('audit_status')}.",
    ]
    if top_feature:
        summary.append(f"Feature mais importante no modelo: {top_feature}.")
    if generalization["warning"]:
        summary.append(
            "Alerta: houve degradacao relevante de PR-AUC fora da validacao."
        )

    return {
        "run_id": run["run_id"],
        "generated_at": datetime.now(timezone.utc),
        "executive_summary": summary,
        "run": {
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "duration_seconds": run.get("duration_seconds"),
            "code_version": run.get("code_version"),
            "dataset_version": run.get("dataset_version"),
            "feature_set_version": run.get("feature_set_version"),
            "experiment_fingerprint": run.get("experiment_fingerprint"),
            "run_directory": run.get("run_directory"),
        },
        "model": {
            "name": run.get("model_name"),
            "parameters": metadata.get("model_params", {}),
            "selected_threshold": run.get("selected_threshold"),
            "threshold_strategy": run.get("threshold_strategy"),
            "selection_engine": run.get("model_selection_engine"),
            "selection_objective": run.get("model_selection_objective"),
            "trial_count": run.get("model_selection_trial_count"),
            "models_considered": sorted(model_names),
            "test_pr_auc_rank": run.get("test_pr_auc_rank"),
            "model_test_pr_auc_rank": run.get("model_test_pr_auc_rank"),
        },
        "dataset": {
            "training_max_rows": run.get("training_max_rows"),
            "splits": {
                split: {
                    "rows": run.get(f"{split}_rows"),
                    "positive_rate": run.get(f"{split}_positive_rate"),
                }
                for split in ("train", "validation", "test", "out_of_time")
            },
            "time_column": metadata.get("time_column"),
            "split_boundaries": audit.get("split_boundaries", {}),
        },
        "performance": {
            "metrics_by_split": metrics,
            "generalization": generalization,
            "cost_configuration": {
                "false_positive_cost": run.get("false_positive_cost"),
                "false_negative_cost": run.get("false_negative_cost"),
            },
            "metric_guide": {
                "pr_auc": "Qualidade de ranking indicada para classes muito desbalanceadas; maior e melhor.",
                "recall": "Proporcao das fraudes reais detectadas.",
                "precision": "Proporcao dos alertas que realmente eram fraude.",
                "fbeta": "Media harmonica com peso configurado maior para recall.",
                "alert_rate": "Fracao das transacoes encaminhada para alerta.",
                "business_cost": "Custo relativo calculado com FP e FN; menor e melhor.",
            },
        },
        "features": {
            "selected_input_columns": selected_columns,
            "selected_input_count": len(selected_columns),
            "transformed_feature_count": data["feature_count"],
            "returned_feature_count": len(features),
            "top_features": features,
            "excluded_input_columns": excluded_columns,
            "excluded_input_columns_source": (
                "persisted_audit"
                if audit.get("excluded_input_columns") is not None
                else "legacy_run_policy_only"
            ),
            "exclusion_policy": {
                "raw_identifiers": list(RAW_ID_HINTS),
                "sensitive": sorted(SENSITIVE_FEATURES),
                "snapshot_when_strict": sorted(SNAPSHOT_FEATURES),
                "post_event_when_strict": sorted(POST_EVENT_FEATURES),
                "date_patterns": ["date", "timestamp", "datetime", "expires", "acct_open"],
                "strict_leakage_prevention": run.get("strict_leakage_prevention"),
            },
            "risk_columns_detected": risk_columns,
            "importance_note": (
                "Importancia representa a contribuicao interna do modelo. "
                "Nao demonstra causalidade e valores entre familias de modelos "
                "nao sao diretamente comparaveis."
            ),
            "exclusion_note": (
                "Runs novos registram a lista exata de colunas descartadas. "
                "Para runs legados sem esse campo, consulte exclusion_policy e "
                "risk_columns_detected."
            ),
        },
        "audit": {
            "status": run.get("audit_status"),
            "warning_count": run.get("audit_warning_count"),
            "failure_count": run.get("audit_failure_count"),
            "checks": data["audit_checks"] or audit.get("check_results", []),
            "warnings": audit.get("warnings", []),
            "failures": audit.get("failures", []),
            "recommendations": audit.get("recommendations", []),
        },
        "model_search": {
            "trials": data["search_trials"],
            "best_trial": data["search_trials"][0] if data["search_trials"] else None,
        },
        "external_benchmarks": data["benchmarks"],
        "robustness_experiments": data["robustness"],
        "threshold_analysis": {
            "evaluation_count": run.get("threshold_evaluation_count"),
            "evaluations": run.get("threshold_evaluations") or [],
        },
        "artifacts": {
            "count": run.get("artifact_count"),
            "total_size_bytes": run.get("artifact_total_size_bytes"),
            "items": run.get("artifacts") or [],
        },
        "baseline": {
            "decision": run.get("promotion_decision"),
            "reasons": run.get("promotion_reason") or [],
            "is_active": run.get("is_active_baseline"),
            "promotion_count": run.get("baseline_promotion_count"),
            "last_promoted_at": run.get("last_promoted_at"),
        },
    }


def _difference(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None:
        return None
    return reference - value


def _relative_drop(reference: float | None, value: float | None) -> float | None:
    if reference in {None, 0} or value is None:
        return None
    return (reference - value) / reference


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"
