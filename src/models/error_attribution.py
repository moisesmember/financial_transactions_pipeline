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
    markdown_path: Path | None = None,
    by_group_path: Path | None = None,
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
    train_cities = set()
    if "train" in split_frames and "merchant_city" in split_frames["train"].columns:
        train_cities = set(
            split_frames["train"]["merchant_city"].dropna().astype(str).str.strip().str.lower()
        )
    for split, frame in split_frames.items():
        y = np.asarray(targets[split], dtype=int)
        split_scores = np.asarray(scores[split], dtype=float)
        predicted = (split_scores >= threshold).astype(int)
        work = frame.reset_index(drop=True).copy()
        work["_score"] = split_scores
        work["_cohort"] = [COHORTS[(int(actual), int(pred))] for actual, pred in zip(y, predicted)]
        if train_cities and "merchant_city" in work.columns:
            work["_city_seen_in_train"] = (
                work["merchant_city"].astype(str).str.strip().str.lower().isin(train_cities)
            )
        report["splits"][split] = {
            cohort: _cohort_summary(work, cohort, top_n)
            for cohort in ("TP", "FP", "FN", "TN")
        }
        report["splits"][split]["fp_vs_fn"] = _compare_fp_fn(work, top_n)
    report["out_of_time_findings"] = _out_of_time_findings(report.get("splits", {}).get("out_of_time"))
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    if by_group_path is not None:
        _group_table(report).to_csv(by_group_path, index=False)
    if markdown_path is not None:
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _cohort_summary(frame: pd.DataFrame, cohort: str, top_n: int) -> dict[str, Any]:
    subset = frame.loc[frame["_cohort"] == cohort]
    amount = _first_column(frame, ("amount", "amount_abs"))
    hour = _first_column(frame, ("transaction_hour", "hour"))
    day_of_week = _first_column(frame, ("transaction_dayofweek", "dayofweek"))
    score_percentiles = subset["_score"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(len(subset)),
        "score_mean": _safe_number(subset["_score"].mean()),
        "score_median": _safe_number(subset["_score"].median()),
        "amount_mean": _safe_number(subset[amount].mean()) if amount else None,
        "amount_median": _safe_number(subset[amount].median()) if amount else None,
        "use_chip_distribution": _distribution(subset, "use_chip", top_n),
        "score_percentiles": {
            f"p{int(quantile * 100):02d}": _safe_number(value)
            for quantile, value in score_percentiles.items()
        },
        "top_mcc": _distribution(subset, "mcc", top_n),
        "top_mcc_description": _distribution(subset, "mcc_description", top_n),
        "top_merchant_city": _distribution(subset, "merchant_city", top_n),
        "top_merchant_state": _distribution(subset, "merchant_state", top_n),
        "transaction_hour_distribution": _distribution(subset, hour, top_n),
        "transaction_dayofweek_distribution": _distribution(subset, day_of_week, top_n),
        "top_features": _numeric_differences(frame, subset, top_n),
        "unseen_city_count": (
            int((~subset["_city_seen_in_train"]).sum())
            if "_city_seen_in_train" in subset.columns
            else None
        ),
        "unseen_city_share": (
            float((~subset["_city_seen_in_train"]).mean())
            if "_city_seen_in_train" in subset.columns and not subset.empty
            else None
        ),
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


def _group_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, groups in report.get("splits", {}).items():
        for cohort in ("TP", "FN", "FP", "TN"):
            summary = groups.get(cohort, {})
            rows.append(
                {
                    "split": split,
                    "group": cohort,
                    "count": summary.get("count"),
                    "amount_mean": summary.get("amount_mean"),
                    "amount_median": summary.get("amount_median"),
                    "score_mean": summary.get("score_mean"),
                    "score_median": summary.get("score_median"),
                    "unseen_city_count": summary.get("unseen_city_count"),
                    "unseen_city_share": summary.get("unseen_city_share"),
                    **{
                        key: value
                        for key, value in (summary.get("score_percentiles") or {}).items()
                    },
                    "use_chip_distribution": json.dumps(summary.get("use_chip_distribution", [])),
                    "top_mcc": json.dumps(summary.get("top_mcc", [])),
                    "top_mcc_description": json.dumps(summary.get("top_mcc_description", [])),
                    "top_merchant_city": json.dumps(summary.get("top_merchant_city", [])),
                    "top_merchant_state": json.dumps(summary.get("top_merchant_state", [])),
                    "transaction_hour_distribution": json.dumps(
                        summary.get("transaction_hour_distribution", [])
                    ),
                    "transaction_dayofweek_distribution": json.dumps(
                        summary.get("transaction_dayofweek_distribution", [])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _out_of_time_findings(groups: dict[str, Any] | None) -> list[str]:
    if not groups:
        return ["Out-of-time attribution is not available."]
    tp = groups.get("TP", {})
    fn = groups.get("FN", {})
    findings = []
    tp_score = tp.get("score_mean")
    fn_score = fn.get("score_mean")
    if tp_score is not None and fn_score is not None:
        findings.append(
            f"FN mean score ({fn_score:.6f}) versus TP mean score ({tp_score:.6f}); "
            "a large gap indicates ranking failure for missed frauds."
        )
    tp_amount = tp.get("amount_median")
    fn_amount = fn.get("amount_median")
    if tp_amount is not None and fn_amount is not None:
        direction = "lower" if fn_amount < tp_amount else "higher"
        findings.append(
            f"FN median amount is {direction} than TP ({fn_amount:.2f} versus {tp_amount:.2f})."
        )
    findings.extend(
        [
            "Compare FN/TP use_chip distributions to identify online versus in-person concentration.",
            "Compare FN MCC, city/state, hour and day-of-week distributions in error_attribution_by_group.csv.",
            f"FN unseen-city share versus training vocabulary: {fn.get('unseen_city_share')}",
        ]
    )
    return findings


def _markdown(report: dict[str, Any]) -> str:
    oot = report.get("splits", {}).get("out_of_time", {})
    lines = [
        "# Error Attribution Report",
        "",
        f"- Threshold: {report.get('threshold')}",
        "- Focus: out-of-time false negatives.",
        "",
        "## Out-of-Time Cohorts",
        "",
        "| Group | Count | Mean score | Median score | Mean amount | Median amount |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort in ("TP", "FN", "FP", "TN"):
        item = oot.get(cohort, {})
        lines.append(
            f"| {cohort} | {item.get('count', 0)} | {item.get('score_mean')} | "
            f"{item.get('score_median')} | {item.get('amount_mean')} | {item.get('amount_median')} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            *[f"- {finding}" for finding in report.get("out_of_time_findings", [])],
            "",
            "Detailed distributions for chip use, MCC, city/state, hour and day of week are available in `error_attribution_by_group.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"
