"""Auditable temporal objective for highly imbalanced fraud models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


INELIGIBLE_OBJECTIVE_SCORE = -1_000_000_000.0


def _finite(values: Iterable[Any]) -> np.ndarray:
    result = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            result.append(numeric)
    return np.asarray(result, dtype=float)


def _stat(values: np.ndarray, operation: str) -> float | None:
    if not len(values):
        return None
    functions = {
        "mean": np.mean,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "std": lambda data: np.std(data, ddof=0),
    }
    return float(functions[operation](values))


def _relative_drop(values: np.ndarray) -> float | None:
    if not len(values):
        return None
    best = float(np.max(values))
    worst = float(np.min(values))
    return (best - worst) / best if best > 0 else 0.0


def temporal_robustness_score(
    window_results: list[dict[str, Any]],
    *,
    min_valid_fold_count: int = 3,
    min_fold_recall_required: float = 0.05,
    min_last_fold_recall_required: float = 0.05,
    min_pr_auc_lift_required: float = 1.25,
    max_alert_rate: float = 0.025,
    max_pr_auc_temporal_drop: float = 0.80,
    max_recall_temporal_drop: float = 0.80,
    false_negative_cost: float = 25.0,
) -> dict[str, Any]:
    """Require minimum quality first, then compare stability.

    PR-AUC is converted to lift against each fold's own prevalence. Undefined
    single-class metrics never enter an average, range, or stability reward.
    """
    no_positive_count = sum(
        item.get("evaluation_status") == "no_positive_class"
        or item.get("metrics", {}).get("evaluation_status") == "no_positive_class"
        for item in window_results
    )
    no_negative_count = sum(
        item.get("evaluation_status") == "no_negative_class"
        or item.get("metrics", {}).get("evaluation_status") == "no_negative_class"
        for item in window_results
    )
    single_class_count = sum(
        (
            item.get("evaluation_status")
            or item.get("metrics", {}).get("evaluation_status")
        )
        in {"no_positive_class", "no_negative_class"}
        for item in window_results
    )

    valid: list[dict[str, Any]] = []
    for item in window_results:
        metrics = item.get("metrics") or {}
        pr_auc = item.get("pr_auc", metrics.get("pr_auc"))
        recall = metrics.get("recall")
        prevalence = item.get("fraud_rate")
        status = item.get("evaluation_status", metrics.get("evaluation_status", "valid"))
        if (
            status == "valid"
            and pr_auc is not None
            and recall is not None
            and prevalence is not None
            and float(prevalence) > 0
        ):
            row = dict(item)
            row["pr_auc"] = float(pr_auc)
            row["pr_auc_lift"] = float(pr_auc) / float(prevalence)
            row["metrics"] = metrics
            valid.append(row)

    pr_aucs = _finite(item["pr_auc"] for item in valid)
    recalls = _finite(item["metrics"].get("recall") for item in valid)
    precisions = _finite(item["metrics"].get("precision") for item in valid)
    alert_rates = _finite(item["metrics"].get("alert_rate") for item in valid)
    costs = _finite(item["metrics"].get("business_cost") for item in valid)
    cost_per_record = _finite(
        item["metrics"].get("cost_per_record")
        if item["metrics"].get("cost_per_record") is not None
        else (
            float(item["metrics"].get("business_cost", 0.0))
            / max(1, int(item.get("row_count", item["metrics"].get("row_count", 0))))
        )
        for item in valid
    )
    lifts = _finite(item["pr_auc_lift"] for item in valid)
    normalized_lifts = np.minimum(1.0, lifts / 10.0) if len(lifts) else lifts

    last = valid[-1] if valid else None
    last_metrics = last["metrics"] if last else {}
    validation = window_results[-1] if window_results else {}
    validation_metrics = validation.get("metrics") or {}

    pr_auc_drop = _relative_drop(lifts)
    recall_drop = _relative_drop(recalls)
    pr_auc_range = (
        float(np.max(pr_aucs) - np.min(pr_aucs)) if len(pr_aucs) else None
    )
    recall_range = (
        float(np.max(recalls) - np.min(recalls)) if len(recalls) else None
    )
    pr_auc_instability_penalty = 0.0
    recall_instability_penalty = 0.0
    if len(normalized_lifts):
        pr_auc_instability_penalty = float(
            0.10 * (np.max(normalized_lifts) - np.min(normalized_lifts))
            + 0.05 * np.std(normalized_lifts, ddof=0)
            + 0.05 * (pr_auc_drop or 0.0)
        )
    if len(recalls):
        recall_instability_penalty = float(
            0.10 * (np.max(recalls) - np.min(recalls))
            + 0.05 * np.std(recalls, ddof=0)
            + 0.05 * (recall_drop or 0.0)
        )
    temporal_instability_penalty = (
        pr_auc_instability_penalty + recall_instability_penalty
    )
    max_observed_alert_rate = _stat(alert_rates, "max")
    alert_rate_penalty = 4.0 * max(
        0.0, (max_observed_alert_rate or 0.0) - max_alert_rate
    )
    mean_cost_per_record = _stat(cost_per_record, "mean")
    cost_penalty = 0.05 * min(
        1.0,
        (mean_cost_per_record or 0.0) / max(float(false_negative_cost), 1e-9),
    )

    min_lift = _stat(lifts, "min")
    last_lift = float(last["pr_auc_lift"]) if last else None
    median_lift = _stat(lifts, "median")
    min_recall = _stat(recalls, "min")
    last_recall = float(last_metrics["recall"]) if last_metrics.get("recall") is not None else None
    median_recall = _stat(recalls, "median")

    quality_score = 0.0
    if valid:
        quality_score = float(
            0.30 * min(1.0, (min_lift or 0.0) / 10.0)
            + 0.20 * (min_recall or 0.0)
            + 0.15 * (last_recall or 0.0)
            + 0.15 * min(1.0, (last_lift or 0.0) / 10.0)
            + 0.10 * (median_recall or 0.0)
            + 0.10 * min(1.0, (median_lift or 0.0) / 10.0)
        )
    total_penalty = float(
        temporal_instability_penalty + alert_rate_penalty + cost_penalty
    )
    raw_score = float(quality_score - total_penalty)

    reasons: list[str] = []
    if len(valid) < min_valid_fold_count:
        reasons.append(
            f"valid_fold_count={len(valid)} below required={min_valid_fold_count}"
        )
    if min_recall is None or min_recall < min_fold_recall_required:
        reasons.append("min_fold_recall below required minimum")
    if last_recall is None or last_recall < min_last_fold_recall_required:
        reasons.append("last_fold_recall below required minimum")
    if min_lift is None or min_lift < min_pr_auc_lift_required:
        reasons.append("min_fold_pr_auc_lift below random-baseline requirement")
    if last_lift is None or last_lift < min_pr_auc_lift_required:
        reasons.append("last_fold_pr_auc_lift below random-baseline requirement")
    if max_observed_alert_rate is not None and max_observed_alert_rate > max_alert_rate:
        reasons.append("max_fold_alert_rate above operational limit")
    if pr_auc_drop is None or pr_auc_drop > max_pr_auc_temporal_drop:
        reasons.append("pr_auc temporal drop above limit")
    if recall_drop is None or recall_drop > max_recall_temporal_drop:
        reasons.append("recall temporal drop above limit")

    eligible = not reasons
    final_score = raw_score if eligible else INELIGIBLE_OBJECTIVE_SCORE
    return {
        "validation_pr_auc": validation.get("pr_auc", validation_metrics.get("pr_auc")),
        "validation_recall": validation_metrics.get("recall"),
        "validation_precision": validation_metrics.get("precision"),
        "validation_alert_rate": validation_metrics.get("alert_rate"),
        "validation_business_cost": validation_metrics.get("business_cost"),
        "mean_fold_pr_auc": _stat(pr_aucs, "mean"),
        "median_fold_pr_auc": _stat(pr_aucs, "median"),
        "min_fold_pr_auc": _stat(pr_aucs, "min"),
        "max_fold_pr_auc": _stat(pr_aucs, "max"),
        "std_fold_pr_auc": _stat(pr_aucs, "std"),
        "last_fold_pr_auc": float(last["pr_auc"]) if last else None,
        "mean_fold_recall": _stat(recalls, "mean"),
        "median_fold_recall": median_recall,
        "min_fold_recall": min_recall,
        "max_fold_recall": _stat(recalls, "max"),
        "std_fold_recall": _stat(recalls, "std"),
        "last_fold_recall": last_recall,
        "mean_fold_precision": _stat(precisions, "mean"),
        "min_fold_precision": _stat(precisions, "min"),
        "last_fold_precision": last_metrics.get("precision"),
        "max_fold_alert_rate": max_observed_alert_rate,
        "mean_fold_alert_rate": _stat(alert_rates, "mean"),
        "last_fold_alert_rate": last_metrics.get("alert_rate"),
        "mean_fold_business_cost": _stat(costs, "mean"),
        "pr_auc_temporal_range": pr_auc_range,
        "recall_temporal_range": recall_range,
        "pr_auc_temporal_drop": pr_auc_drop,
        "recall_temporal_drop": recall_drop,
        "pr_auc_instability_penalty": pr_auc_instability_penalty,
        "recall_instability_penalty": recall_instability_penalty,
        "temporal_instability_penalty": temporal_instability_penalty,
        "alert_rate_penalty": alert_rate_penalty,
        "cost_penalty": cost_penalty,
        "total_penalty": total_penalty,
        "random_pr_auc_baseline": _stat(
            _finite(item.get("fraud_rate") for item in valid), "median"
        ),
        "min_fold_pr_auc_lift": min_lift,
        "last_fold_pr_auc_lift": last_lift,
        "median_fold_pr_auc_lift": median_lift,
        "normalized_min_fold_pr_auc_lift": (
            min(1.0, (min_lift or 0.0) / 10.0) if min_lift is not None else None
        ),
        "normalized_last_fold_pr_auc_lift": (
            min(1.0, (last_lift or 0.0) / 10.0) if last_lift is not None else None
        ),
        "normalized_median_fold_pr_auc_lift": (
            min(1.0, (median_lift or 0.0) / 10.0) if median_lift is not None else None
        ),
        "valid_fold_count": len(valid),
        "single_class_fold_count": int(single_class_count),
        "no_positive_period_count": int(no_positive_count),
        "no_negative_period_count": int(no_negative_count),
        "eligibility_status": "eligible" if eligible else "ineligible",
        "ineligibility_reasons": reasons,
        "quality_score": quality_score,
        "raw_temporal_score": raw_score,
        "final_objective_score": final_score,
        "temporal_robustness_score": final_score,
    }


def write_objective_breakdown(
    rows: list[dict[str, Any]], csv_path: Path, markdown_path: Path
) -> None:
    """Persist a machine-readable and human-readable trial score audit."""
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    lines = [
        "# Objective Score Breakdown",
        "",
        "Quality is checked before stability. Ineligible trials receive the fixed ",
        f"objective floor `{INELIGIBLE_OBJECTIVE_SCORE:g}` and cannot win.",
        "",
        "## Formula",
        "",
        "`quality = 0.30*min_lift_norm + 0.20*min_recall + 0.15*last_recall + "
        "0.15*last_lift_norm + 0.10*median_recall + 0.10*median_lift_norm`",
        "",
        "`raw_score = quality - (PR-AUC instability + recall instability + "
        "alert-rate penalty + cost penalty)`",
        "",
        "| Trial | Model | Eligible | Quality | Penalty | Raw score | Final score | Reasons |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        reasons = row.get("ineligibility_reasons", [])
        if isinstance(reasons, str):
            try:
                reasons = json.loads(reasons)
            except json.JSONDecodeError:
                reasons = [reasons]
        lines.append(
            f"| {row.get('trial_number')} | {row.get('model_name')} | "
            f"{row.get('eligibility_status')} | {float(row.get('quality_score') or 0):.6f} | "
            f"{float(row.get('total_penalty') or 0):.6f} | "
            f"{float(row.get('raw_temporal_score') or 0):.6f} | "
            f"{float(row.get('final_objective_score') if row.get('final_objective_score') is not None else INELIGIBLE_OBJECTIVE_SCORE):.6f} | "
            f"{'<br>'.join(reasons) if reasons else '-'} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
