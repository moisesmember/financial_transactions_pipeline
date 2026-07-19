"""Operational Top-K alert-capacity analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_top_k_analysis(
    split_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    k_values: tuple[int, ...],
    false_positive_cost: float,
    false_negative_cost: float,
    csv_path: Path,
    markdown_path: Path,
) -> pd.DataFrame:
    """Measure fraud recovery when operations investigate exactly the top K scores."""
    rows: list[dict[str, object]] = []
    for split, (target, scores) in split_scores.items():
        y = np.asarray(target, dtype=int)
        probability = np.asarray(scores, dtype=float)
        order = np.argsort(-probability, kind="stable")
        total_positives = int(y.sum())
        for requested_k in k_values:
            effective_k = min(int(requested_k), len(y))
            selected = order[:effective_k]
            frauds_found = int(y[selected].sum())
            false_alerts = effective_k - frauds_found
            missed_frauds = total_positives - frauds_found
            rows.append(
                {
                    "split": split,
                    "strategy": f"top_{requested_k}_alerts",
                    "requested_k": int(requested_k),
                    "effective_k": effective_k,
                    "precision_at_k": frauds_found / effective_k if effective_k else 0.0,
                    "recall_at_k": frauds_found / total_positives if total_positives else 0.0,
                    "frauds_found": frauds_found,
                    "false_alerts": false_alerts,
                    "missed_frauds": missed_frauds,
                    "business_cost": float(
                        false_alerts * false_positive_cost
                        + missed_frauds * false_negative_cost
                    ),
                    "minimum_score_at_k": (
                        float(probability[selected[-1]]) if effective_k else None
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(csv_path, index=False)
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(result: pd.DataFrame) -> str:
    lines = [
        "# Top-K Analysis",
        "",
        "Top-K is an operational capacity analysis; it cannot promote a temporally weak model.",
        "",
        "| Split | Strategy | Precision@K | Recall@K | Frauds found | False alerts | Cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples():
        lines.append(
            f"| {row.split} | {row.strategy} | {row.precision_at_k:.6f} | "
            f"{row.recall_at_k:.6f} | {row.frauds_found} | {row.false_alerts} | "
            f"{row.business_cost:.2f} |"
        )
    return "\n".join(lines) + "\n"
