"""Threshold comparison and business-cost selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Build an inclusive threshold grid with stable decimal values."""
    if not 0 <= start <= stop <= 1:
        raise ValueError("Thresholds devem respeitar 0 <= inicio <= fim <= 1.")
    if step <= 0:
        raise ValueError("O passo do threshold deve ser positivo.")
    count = int(np.floor((stop - start) / step)) + 1
    values = start + np.arange(count) * step
    if values[-1] < stop and not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.round(values, 10)


def build_threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    beta: float,
    false_positive_cost: float,
    false_negative_cost: float,
    split: str,
    scenario_name: str = "primary",
) -> pd.DataFrame:
    """Compare classification outcomes and business cost across thresholds."""
    rows: list[dict[str, float | str]] = []
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sample_count = len(y_true)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        business_cost = (
            fp * false_positive_cost
            + fn * false_negative_cost
        )
        rows.append(
            {
                "scenario_name": scenario_name,
                "split": split,
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "fbeta": float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0)),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "alerts": int(tp + fp),
                "alert_rate": float((tp + fp) / sample_count),
                "business_cost": float(business_cost),
                "cost_per_record": float(business_cost / sample_count),
                "false_positive_cost": float(false_positive_cost),
                "false_negative_cost": float(false_negative_cost),
            }
        )
    return pd.DataFrame(rows)


def select_business_threshold(table: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Select the lowest-cost validation threshold with deterministic tie-breaks."""
    validation = table.loc[table["split"].eq("validation")]
    if validation.empty:
        raise ValueError("A tabela deve conter thresholds do split de validacao.")

    selected = validation.sort_values(
        ["business_cost", "fn", "fp", "threshold"],
        ascending=[True, True, True, False],
    ).iloc[0]
    metrics = {
        key: float(selected[key])
        for key in (
            "business_cost",
            "cost_per_record",
            "precision",
            "recall",
            "f1",
            "fbeta",
            "tp",
            "fp",
            "tn",
            "fn",
        )
    }
    return float(selected["threshold"]), metrics


def build_cost_scenario_summary(
    split_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    beta: float,
    cost_scenarios: tuple[tuple[float, float], ...],
) -> pd.DataFrame:
    """Select a validation threshold for each cost scenario and evaluate every split."""
    rows: list[dict[str, float | str]] = []
    for false_positive_cost, false_negative_cost in cost_scenarios:
        scenario_name = f"fp_{false_positive_cost:g}_fn_{false_negative_cost:g}"
        tables = {
            split: build_threshold_table(
                y_true,
                y_score,
                thresholds=thresholds,
                beta=beta,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
                split=split,
                scenario_name=scenario_name,
            )
            for split, (y_true, y_score) in split_scores.items()
        }
        selected_threshold, _ = select_business_threshold(tables["validation"])
        for split, table in tables.items():
            selected = table.loc[np.isclose(table["threshold"], selected_threshold)].iloc[0]
            rows.append(selected.to_dict())
    return pd.DataFrame(rows)
