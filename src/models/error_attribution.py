"""Error cohort attribution for governed fraud model reviews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COHORTS = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}


def build_error_attribution_report(
    split_frames: dict[str, pd.DataFrame],
    targets: dict[str, pd.Series],
    scores: dict[str, np.ndarray],
    threshold: float,
    output_path: Path,
    top_n: int = 10,
) -> dict[str, Any]:
    """Describe prediction cohorts without using evaluation data for model selection."""
    report: dict[str, Any] = {
        "status": "completed",
        "threshold": float(threshold),
        "attribution_method": "standardized_numeric_cohort_difference",
        "shap_status": "not_computed",
        "shap_note": (
            "SHAP is not computed automatically; top_features are descriptive cohort "
            "differences and must not be interpreted as causal or SHAP attribution."
        ),
        "splits": {},
    }
    for split, frame in split_frames.items():
        y = np.asarray(targets[split], dtype=int)
        split_scores = np.asarray(scores[split], dtype=float)
        predicted = (split_scores >= threshold).astype(int)
        work = frame.reset_index(drop=True).copy()
        work["_score"] = split_scores
        work["_cohort"] = [COHORTS[(int(actual), int(pred))] for actual, pred in zip(y, predicted)]
        report["splits"][split] = {
            cohort: _cohort_summary(work, cohort, top_n)
            for cohort in ("TP", "FP", "FN", "TN")
        }
        report["splits"][split]["fp_vs_fn"] = _compare_fp_fn(work, top_n)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    return report


def _cohort_summary(frame: pd.DataFrame, cohort: str, top_n: int) -> dict[str, Any]:
    subset = frame.loc[frame["_cohort"] == cohort]
    amount = _first_column(frame, ("amount", "amount_abs"))
    hour = _first_column(frame, ("transaction_hour", "hour"))
    return {
        "count": int(len(subset)),
        "score_mean": _safe_number(subset["_score"].mean()),
        "score_median": _safe_number(subset["_score"].median()),
        "amount_mean": _safe_number(subset[amount].mean()) if amount else None,
        "amount_median": _safe_number(subset[amount].median()) if amount else None,
        "use_chip_distribution": _distribution(subset, "use_chip", top_n),
        "top_mcc": _distribution(subset, _first_column(frame, ("mcc_description", "mcc")), top_n),
        "top_merchant_city": _distribution(subset, "merchant_city", top_n),
        "top_merchant_state": _distribution(subset, "merchant_state", top_n),
        "common_hours": _distribution(subset, hour, top_n),
        "top_features": _numeric_differences(frame, subset, top_n),
    }


def _compare_fp_fn(frame: pd.DataFrame, top_n: int) -> dict[str, Any]:
    fp = frame.loc[frame["_cohort"] == "FP"]
    fn = frame.loc[frame["_cohort"] == "FN"]
    amount = _first_column(frame, ("amount", "amount_abs"))
    return {
        "score_mean_difference_fp_minus_fn": _safe_number(fp["_score"].mean() - fn["_score"].mean()),
        "amount_mean_difference_fp_minus_fn": (
            _safe_number(fp[amount].mean() - fn[amount].mean()) if amount else None
        ),
        "largest_numeric_differences": _pairwise_numeric_differences(fp, fn, top_n),
        "use_chip_fp": _distribution(fp, "use_chip", top_n),
        "use_chip_fn": _distribution(fn, "use_chip", top_n),
    }


def _numeric_differences(all_rows: pd.DataFrame, cohort: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    numeric = [column for column in all_rows.select_dtypes(include="number").columns if not column.startswith("_")]
    rows = []
    for column in numeric:
        scale = float(all_rows[column].std())
        if not np.isfinite(scale) or scale == 0:
            continue
        difference = (float(cohort[column].mean()) - float(all_rows[column].mean())) / scale
        if np.isfinite(difference):
            rows.append({"feature": column, "standardized_difference": difference})
    return sorted(rows, key=lambda item: abs(item["standardized_difference"]), reverse=True)[:top_n]


def _pairwise_numeric_differences(left: pd.DataFrame, right: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    numeric = [column for column in left.select_dtypes(include="number").columns if not column.startswith("_")]
    rows = []
    for column in numeric:
        difference = float(left[column].mean()) - float(right[column].mean())
        if np.isfinite(difference):
            rows.append({"feature": column, "mean_difference_fp_minus_fn": difference})
    return sorted(rows, key=lambda item: abs(item["mean_difference_fp_minus_fn"]), reverse=True)[:top_n]


def _distribution(frame: pd.DataFrame, column: str | None, top_n: int) -> list[dict[str, Any]]:
    if not column or column not in frame or frame.empty:
        return []
    counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False).head(top_n)
    total = max(1, len(frame))
    return [{"value": value, "count": int(count), "share": float(count / total)} for value, count in counts.items()]


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _safe_number(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
