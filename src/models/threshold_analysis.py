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


def build_threshold_recommendations(
    threshold_table: pd.DataFrame,
    max_alert_rate: float,
) -> dict[str, dict[str, float | str] | None]:
    """Generate threshold recommendations without changing the validation-selected threshold."""
    validation = threshold_table.loc[threshold_table["split"].eq("validation")]
    if validation.empty:
        return {
            "validation_lowest_cost": None,
            "most_stable": None,
            "max_recall_with_alert_cap": None,
            "retrospective_out_of_time_lowest_cost": None,
        }
    validation_lowest_cost = _row_to_dict(
        validation.sort_values(
            ["business_cost", "fn", "fp", "threshold"],
            ascending=[True, True, True, False],
        ).iloc[0],
        rationale="Menor custo na validacao; elegivel para selecao operacional.",
    )

    stable = _stable_threshold(threshold_table)
    alert_cap = validation.loc[validation["alert_rate"].le(max_alert_rate)]
    max_recall_with_alert_cap = None
    if not alert_cap.empty:
        max_recall_with_alert_cap = _row_to_dict(
            alert_cap.sort_values(
                ["recall", "business_cost", "threshold"],
                ascending=[False, True, False],
            ).iloc[0],
            rationale="Maior recall na validacao respeitando capacidade operacional.",
        )

    oot = threshold_table.loc[threshold_table["split"].eq("out_of_time")]
    retrospective_oot = None
    if not oot.empty:
        retrospective_oot = _row_to_dict(
            oot.sort_values(
                ["business_cost", "fn", "fp", "threshold"],
                ascending=[True, True, True, False],
            ).iloc[0],
            rationale="Menor custo OOT apenas retrospectivo; nao usar para selecao real.",
        )

    return {
        "validation_lowest_cost": validation_lowest_cost,
        "most_stable": stable,
        "max_recall_with_alert_cap": max_recall_with_alert_cap,
        "retrospective_out_of_time_lowest_cost": retrospective_oot,
    }


def _stable_threshold(threshold_table: pd.DataFrame) -> dict[str, float | str] | None:
    required_splits = {"validation", "test", "out_of_time"}
    if not required_splits <= set(threshold_table["split"]):
        return None
    rows = []
    for threshold, group in threshold_table.groupby("threshold"):
        if not required_splits <= set(group["split"]):
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "split": "all",
                "precision": float(group["precision"].mean()),
                "recall": float(group["recall"].mean()),
                "f1": float(group["f1"].mean()),
                "fbeta": float(group["fbeta"].mean()),
                "alert_rate": float(group["alert_rate"].mean()),
                "business_cost": float(group["business_cost"].mean()),
                "cost_per_record": float(group["cost_per_record"].mean()),
                "stability_score": float(
                    group["recall"].std(ddof=0)
                    + group["cost_per_record"].std(ddof=0)
                    + group["alert_rate"].std(ddof=0)
                ),
            }
        )
    if not rows:
        return None
    selected = pd.DataFrame(rows).sort_values(
        ["stability_score", "business_cost", "threshold"],
        ascending=[True, True, False],
    ).iloc[0]
    return _row_to_dict(
        selected,
        rationale="Menor variacao de recall, custo por registro e alert rate entre splits.",
    )


def _row_to_dict(row: pd.Series, rationale: str) -> dict[str, float | str]:
    keys = (
        "split",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fbeta",
        "alert_rate",
        "business_cost",
        "cost_per_record",
        "stability_score",
    )
    payload = {
        key: float(row[key]) if key != "split" else str(row[key])
        for key in keys
        if key in row and pd.notna(row[key])
    }
    payload["rationale"] = rationale
    return payload
