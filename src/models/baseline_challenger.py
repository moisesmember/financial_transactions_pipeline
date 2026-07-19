"""Retrospective baseline/challenger comparison without tuning on final holdouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_baseline_challenger(
    baseline: dict[str, Any] | None,
    challenger: dict[str, Any],
) -> dict[str, Any]:
    """Return a conservative replacement decision across all available periods."""
    baseline = baseline or {}
    comparisons: dict[str, Any] = {}
    reasons: list[str] = []
    required_splits = ("validation", "test", "out_of_time")
    for split in required_splits:
        baseline_metrics = baseline.get(f"{split}_metrics") or {}
        challenger_metrics = challenger.get(f"{split}_metrics") or {}
        if not baseline_metrics:
            reasons.append(f"baseline metrics unavailable for {split}")
            comparisons[split] = {"status": "unavailable"}
            continue
        baseline_rate = _positive_rate(baseline, split)
        challenger_rate = _positive_rate(challenger, split)
        baseline_cost = _business_cost(baseline_metrics)
        challenger_cost = _business_cost(challenger_metrics)
        checks = {
            "pr_auc_not_worse": _ge(challenger_metrics.get("pr_auc"), baseline_metrics.get("pr_auc")),
            "recall_not_worse": _ge(challenger_metrics.get("recall"), baseline_metrics.get("recall")),
            "precision_not_worse": _ge(challenger_metrics.get("precision"), baseline_metrics.get("precision")),
            "false_positives_not_higher": _le(challenger_metrics.get("fp"), baseline_metrics.get("fp")),
            "business_cost_not_higher": _le(challenger_cost, baseline_cost),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            reasons.append(f"{split}: " + ", ".join(failed))
        comparisons[split] = {
            "status": "pass" if not failed else "fail",
            "checks": checks,
            "baseline": _metric_payload(baseline_metrics, baseline_rate),
            "challenger": _metric_payload(challenger_metrics, challenger_rate),
        }

    baseline_temporal = (baseline.get("model_selection") or {}).get("best_trial_breakdown", {})
    challenger_temporal = (challenger.get("model_selection") or {}).get(
        "best_trial_breakdown", {}
    )
    for metric in (
        "min_fold_pr_auc_lift",
        "min_fold_recall",
        "last_fold_pr_auc_lift",
        "last_fold_recall",
    ):
        baseline_value = baseline_temporal.get(metric)
        challenger_value = challenger_temporal.get(metric)
        if baseline_value is not None and not _ge(challenger_value, baseline_value):
            reasons.append(f"temporal: {metric} worse than baseline")

    replace = bool(baseline) and not reasons
    return {
        "best_rejected_baseline": {
            "model_name": baseline.get("model_name", "hist_gradient_boosting"),
            "role": "best_rejected_baseline",
            "decision": (baseline.get("baseline_decision") or {}).get("decision", "reject"),
        },
        "challenger": {
            "model_name": challenger.get("model_name", "logistic_regression_regularized"),
            "role": "rejected_challenger" if not replace else "eligible_challenger",
            "decision": (challenger.get("baseline_decision") or {}).get("decision", "reject"),
        },
        "comparisons": comparisons,
        "temporal": {"baseline": baseline_temporal, "challenger": challenger_temporal},
        "challenger_replaces_baseline": replace,
        "decision": "replace" if replace else "keep_best_rejected_baseline",
        "reasons": reasons or ["challenger improved consistently across all required gates"],
        "retrospective_holdout_use_only": True,
        "test_or_oot_used_by_optuna": False,
    }


def write_baseline_challenger_comparison(
    baseline: dict[str, Any] | None,
    challenger: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    payload = compare_baseline_challenger(baseline, challenger)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    lines = [
        "# Baseline / Challenger Comparison",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Baseline: `{payload['best_rejected_baseline']['model_name']}` (`best_rejected_baseline`)",
        f"- Challenger: `{payload['challenger']['model_name']}` (`{payload['challenger']['role']}`)",
        "- Test and out-of-time are retrospective final gates, never Optuna inputs.",
        "",
        "## Reasons",
        "",
        *[f"- {reason}" for reason in payload["reasons"]],
        "",
        "## Split checks",
        "",
        "| Split | Status | Baseline PR-AUC | Challenger PR-AUC | Baseline recall | Challenger recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, comparison in payload["comparisons"].items():
        baseline_metrics = comparison.get("baseline", {})
        challenger_metrics = comparison.get("challenger", {})
        lines.append(
            f"| {split} | {comparison['status']} | {baseline_metrics.get('pr_auc')} | "
            f"{challenger_metrics.get('pr_auc')} | {baseline_metrics.get('recall')} | "
            f"{challenger_metrics.get('recall')} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _metric_payload(metrics: dict[str, Any], positive_rate: float | None) -> dict[str, Any]:
    pr_auc = metrics.get("pr_auc")
    return {
        key: metrics.get(key)
        for key in ("pr_auc", "precision", "recall", "tp", "fp", "fn", "alert_rate")
    } | {
        "pr_auc_lift": (
            float(pr_auc) / positive_rate
            if pr_auc is not None and positive_rate is not None and positive_rate > 0
            else None
        ),
        "business_cost": _business_cost(metrics),
    }


def _positive_rate(metadata: dict[str, Any], split: str) -> float | None:
    value = (metadata.get("dataset") or {}).get(f"{split}_positive_rate")
    return float(value) if value is not None else None


def _business_cost(metrics: dict[str, Any]) -> float | None:
    if metrics.get("business_cost") is not None:
        return float(metrics["business_cost"])
    if metrics.get("fp") is None or metrics.get("fn") is None:
        return None
    return float(metrics["fp"]) + 25.0 * float(metrics["fn"])


def _ge(left: Any, right: Any) -> bool:
    return left is not None and right is not None and float(left) >= float(right)


def _le(left: Any, right: Any) -> bool:
    return left is not None and right is not None and float(left) <= float(right)
