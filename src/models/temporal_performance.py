"""Yearly and monthly performance diagnostics for temporal fraud evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_binary_classifier


def write_temporal_performance_artifacts(
    split_frames: dict[str, pd.DataFrame],
    targets: dict[str, pd.Series],
    scores: dict[str, np.ndarray],
    date_column: str,
    threshold: float,
    beta: float,
    false_positive_cost: float,
    false_negative_cost: float,
    year_path: Path,
    month_path: Path,
    markdown_path: Path,
) -> dict[str, pd.DataFrame]:
    """Evaluate each split independently by calendar year and month."""
    year = _performance_rows(
        split_frames,
        targets,
        scores,
        date_column,
        threshold,
        beta,
        false_positive_cost,
        false_negative_cost,
        frequency="Y",
    )
    month = _performance_rows(
        split_frames,
        targets,
        scores,
        date_column,
        threshold,
        beta,
        false_positive_cost,
        false_negative_cost,
        frequency="M",
    )
    year.to_csv(year_path, index=False)
    month.to_csv(month_path, index=False)
    markdown_path.write_text(_markdown(year, month), encoding="utf-8")
    return {"year": year, "month": month}


def _performance_rows(
    split_frames: dict[str, pd.DataFrame],
    targets: dict[str, pd.Series],
    scores: dict[str, np.ndarray],
    date_column: str,
    threshold: float,
    beta: float,
    false_positive_cost: float,
    false_negative_cost: float,
    frequency: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in split_frames.items():
        if date_column not in frame.columns:
            raise ValueError(f"Coluna temporal ausente em {split}: {date_column}.")
        dates = pd.to_datetime(frame[date_column], errors="coerce").reset_index(drop=True)
        target = pd.Series(targets[split]).astype(int).reset_index(drop=True)
        split_scores = pd.Series(np.asarray(scores[split], dtype=float))
        if not (len(dates) == len(target) == len(split_scores)):
            raise ValueError(f"Frame, target e scores desalinhados no split {split}.")
        work = pd.DataFrame({"date": dates, "target": target, "score": split_scores}).dropna(
            subset=["date"]
        )
        work["period"] = work["date"].dt.to_period(frequency).astype(str)
        for period, group in work.groupby("period", sort=True):
            metrics = evaluate_binary_classifier(
                group["target"].to_numpy(),
                group["score"].to_numpy(),
                threshold=threshold,
                beta=beta,
            )
            positives = group.loc[group["target"].eq(1), "score"]
            rows.append(
                {
                    "split": split,
                    "period": period,
                    "evaluation_status": metrics["evaluation_status"],
                    "rows": int(len(group)),
                    "positive_count": int(group["target"].sum()),
                    "negative_count": int(group["target"].eq(0).sum()),
                    "fraud_rate": float(group["target"].mean()),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "tn": int(metrics["tn"]),
                    "fn": int(metrics["fn"]),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fbeta": metrics["fbeta"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "alert_rate": metrics["alert_rate"],
                    "business_cost": float(
                        metrics["fp"] * false_positive_cost
                        + metrics["fn"] * false_negative_cost
                    ),
                    "mean_score": float(group["score"].mean()),
                    "median_score": float(group["score"].median()),
                    "positive_mean_score": (
                        float(positives.mean()) if not positives.empty else np.nan
                    ),
                    "positive_median_score": (
                        float(positives.median()) if not positives.empty else np.nan
                    ),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(rows)


def _markdown(year: pd.DataFrame, month: pd.DataFrame) -> str:
    evaluation = year.loc[year["split"].isin(["test", "out_of_time"])].copy()
    future_months = month.loc[month["split"].isin(["test", "out_of_time"])].copy()
    lines = [
        "# Performance By Period",
        "",
        "## Executive Findings",
        "",
    ]
    no_positive = int(month["evaluation_status"].eq("no_positive_class").sum())
    no_negative = int(month["evaluation_status"].eq("no_negative_class").sum())
    lines.extend(
        [
            f"- Monthly periods without positives: {no_positive}; PR-AUC, recall and F-beta are undefined.",
            f"- Monthly periods without negatives: {no_negative}; ROC-AUC is undefined.",
        ]
    )
    valid_evaluation = evaluation.loc[evaluation["evaluation_status"].eq("valid")]
    if valid_evaluation.empty:
        lines.append("- No valid two-class test/out-of-time periods are available.")
    else:
        worst = valid_evaluation.sort_values(
            ["recall", "pr_auc", "positive_count"], na_position="last"
        ).iloc[0]
        lines.append(
            f"- Worst future year: `{worst['period']}` ({worst['split']}), "
            f"recall={_format_metric(worst['recall'])}, PR-AUC={_format_metric(worst['pr_auc'])}."
        )
        years_2017_2019 = evaluation.loc[evaluation["period"].isin(["2017", "2018", "2019"])]
        if not years_2017_2019.empty:
            ranked = years_2017_2019.sort_values("recall")
            lines.append(
                "- Recall 2017-2019 (worst first): "
                + ", ".join(
                    f"{row.period}={_format_metric(row.recall)}"
                    for row in ranked.itertuples()
                )
                + "."
            )
        near_zero = evaluation.loc[
            evaluation["positive_count"].gt(0) & evaluation["recall"].le(0.01)
        ]
        lines.append(
            "- Years with practically zero fraud detection: "
            + (", ".join(near_zero["period"].astype(str)) if not near_zero.empty else "none")
            + "."
        )
    if not future_months.empty:
        valid_future_months = future_months.loc[
            future_months["evaluation_status"].eq("valid")
        ].copy()
        worst_months = valid_future_months.sort_values(
            ["recall", "pr_auc"], na_position="last"
        ).head(5)
        lines.extend(
            [
                "- Worst months: "
                + (
                    ", ".join(
                        f"{row.period} ({row.split}, recall={row.recall:.4f})"
                        for row in worst_months.itertuples()
                    )
                    if not worst_months.empty
                    else "none with both classes"
                )
                + ".",
                "- `positive_mean_score` and `positive_median_score` show whether future fraud scores collapsed.",
                "",
                "## Future Periods",
                "",
                "| Split | Period | Rows | Positives | Recall | PR-AUC | Positive mean score | Cost |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in evaluation.sort_values(["period", "split"]).itertuples():
            lines.append(
                f"| {row.split} | {row.period} | {row.rows} | {row.positive_count} | "
                f"{_format_metric(row.recall)} | {_format_metric(row.pr_auc)} | "
                f"{_format_metric(row.positive_mean_score)} | "
                f"{row.business_cost:.2f} |"
            )
    return "\n".join(lines) + "\n"


def _format_metric(value: object) -> str:
    return "null" if value is None or pd.isna(value) else f"{float(value):.6f}"
