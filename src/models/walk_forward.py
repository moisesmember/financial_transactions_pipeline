"""Optional temporal walk-forward validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data.limit_data import TrainingDataLimiter
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer


def summarize_walk_forward_folds(
    completed: list[dict[str, Any]],
    min_recall: float,
    min_pr_auc_lift: float,
    max_recall_drop: float,
) -> dict[str, Any]:
    """Summarize temporal fold stability with emphasis on the last and worst fold."""
    if not completed:
        return {
            "best_fold": None,
            "worst_fold": None,
            "mean_recall": None,
            "median_recall": None,
            "std_recall": None,
            "mean_pr_auc": None,
            "median_pr_auc": None,
            "std_pr_auc": None,
            "min_recall_fold": None,
            "min_pr_auc_fold": None,
            "last_fold_recall": None,
            "last_fold_pr_auc": None,
            "last_fold_penalty": 0.0,
            "recall_drop_best_to_worst": None,
            "pr_auc_drop_best_to_worst": None,
            "unstable": False,
            "failure_reasons": [],
        }
    recalls = np.array([float(row["recall"]) for row in completed], dtype=float)
    pr_aucs = np.array([float(row["pr_auc"]) for row in completed], dtype=float)
    best = max(completed, key=lambda row: (float(row["recall"]), float(row["pr_auc"])))
    worst = min(completed, key=lambda row: (float(row["recall"]), float(row["pr_auc"])))
    min_pr_auc_row = min(completed, key=lambda row: float(row["pr_auc"]))
    last = completed[-1]
    best_recall = float(best["recall"])
    worst_recall = float(worst["recall"])
    best_pr_auc = float(best["pr_auc"])
    worst_pr_auc = float(min_pr_auc_row["pr_auc"])
    last_recall = float(last["recall"])
    last_pr_auc = float(last["pr_auc"])
    last_random_pr_auc = float(last.get("fraud_rate") or 0.0)
    minimum_last_pr_auc = last_random_pr_auc * min_pr_auc_lift
    recall_drop = (best_recall - worst_recall) / best_recall if best_recall > 0 else 0.0
    pr_auc_drop = (best_pr_auc - worst_pr_auc) / best_pr_auc if best_pr_auc > 0 else 0.0
    last_fold_penalty = (
        max(0.0, float(np.median(recalls)) - last_recall)
        + max(0.0, float(np.median(pr_aucs)) - last_pr_auc)
    )
    failure_reasons: list[str] = []
    if worst_recall < min_recall:
        failure_reasons.append("Pior fold com recall abaixo do minimo temporal.")
    if last_recall < min_recall:
        failure_reasons.append("Ultimo fold com recall abaixo do minimo temporal.")
    if last_pr_auc <= minimum_last_pr_auc:
        failure_reasons.append("Ultimo fold com PR-AUC proximo do baseline aleatorio.")
    if recall_drop > max_recall_drop:
        failure_reasons.append("Queda de recall entre melhor e pior fold acima do limite.")
    return {
        "best_fold": int(best["fold"]),
        "worst_fold": int(worst["fold"]),
        "mean_recall": float(recalls.mean()),
        "median_recall": float(np.median(recalls)),
        "std_recall": float(recalls.std(ddof=0)),
        "mean_pr_auc": float(pr_aucs.mean()),
        "median_pr_auc": float(np.median(pr_aucs)),
        "std_pr_auc": float(pr_aucs.std(ddof=0)),
        "min_recall_fold": float(worst_recall),
        "min_pr_auc_fold": float(worst_pr_auc),
        "last_fold_recall": last_recall,
        "last_fold_pr_auc": last_pr_auc,
        "last_fold_random_pr_auc": last_random_pr_auc,
        "last_fold_penalty": float(last_fold_penalty),
        "recall_drop_best_to_worst": float(recall_drop),
        "pr_auc_drop_best_to_worst": float(pr_auc_drop),
        "unstable": bool(failure_reasons),
        "failure_reasons": failure_reasons,
    }


class WalkForwardValidator:
    """Run expanding-window validation folds when enabled."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        frame: pd.DataFrame,
        time_column: str,
        model_name: str,
        model_params: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Write walk-forward JSON/Markdown and return the payload."""
        if not self.settings.walk_forward_enabled:
            return self._write(
                output_dir,
                {
                    "status": "disabled",
                    "warnings": ["Walk-forward desabilitado por configuracao."],
                    "folds": [],
                },
            )
        ordered = frame.sort_values(time_column).reset_index(drop=True)
        folds = self._fold_boundaries(len(ordered), self.settings.walk_forward_folds)
        thresholds = threshold_grid(
            self.settings.threshold_analysis_start,
            self.settings.threshold_analysis_stop,
            self.settings.threshold_analysis_step,
        )
        rows = []
        for fold_number, (train_end, validation_end) in enumerate(folds, start=1):
            train = ordered.iloc[:train_end].copy()
            validation = ordered.iloc[train_end:validation_end].copy()
            if train[self.settings.target_column].nunique() < 2 or validation.empty:
                rows.append(
                    {
                        "fold": fold_number,
                        "status": "skipped",
                        "message": "Fold sem positivos/negativos suficientes.",
                    }
                )
                continue
            train_rows_before_sampling = len(train)
            train_positives_before_sampling = int(
                train[self.settings.target_column].eq(1).sum()
            )
            if self.settings.negative_sampling_enabled:
                train = TrainingDataLimiter(
                    self.settings.training_max_rows,
                    negative_positive_ratio=self.settings.training_negative_positive_ratio,
                    random_state=self.settings.random_state + fold_number,
                    period="M" if self.settings.negative_sampling_by == "month" else "Y",
                ).apply(
                    train,
                    target_column=self.settings.target_column,
                    time_column=time_column,
                )
            started = datetime.now(timezone.utc)
            X_train, y_train = self._split_xy(train)
            X_validation, y_validation = self._split_xy(validation)
            pipeline = FraudModelTrainer(self.settings).train(
                X_train,
                y_train,
                model_name=model_name,
                model_params=model_params,
            )
            scores = pipeline.predict_proba(X_validation)[:, 1]
            table = build_threshold_table(
                y_validation.to_numpy(),
                scores,
                thresholds=thresholds,
                beta=self.settings.threshold_beta,
                false_positive_cost=self.settings.false_positive_cost,
                false_negative_cost=self.settings.false_negative_cost,
                split="validation",
            )
            threshold, _ = select_business_threshold(table)
            metrics = evaluate_binary_classifier(
                y_validation.to_numpy(),
                scores,
                threshold=threshold,
                beta=self.settings.threshold_beta,
            )
            business_cost = (
                metrics["fp"] * self.settings.false_positive_cost
                + metrics["fn"] * self.settings.false_negative_cost
            )
            rows.append(
                {
                    "fold": fold_number,
                    "status": "completed",
                    "train_rows": int(len(train)),
                    "train_rows_before_sampling": int(train_rows_before_sampling),
                    "train_positive_count_before_sampling": train_positives_before_sampling,
                    "train_positive_count_after_sampling": int(y_train.sum()),
                    "validation_rows": int(len(validation)),
                    "validation_time_min": validation[time_column].min().isoformat(),
                    "validation_time_max": validation[time_column].max().isoformat(),
                    "threshold": threshold,
                    "fraud_rate": float(y_validation.mean()),
                    "positive_count": int(y_validation.sum()),
                    "alerts": int(metrics["alerts"]),
                    "business_cost": float(business_cost),
                    "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
                    **metrics,
                }
            )
        warnings = []
        completed = [row for row in rows if row["status"] == "completed"]
        summary = summarize_walk_forward_folds(
            completed,
            min_recall=self.settings.promotion_min_walk_forward_recall,
            min_pr_auc_lift=self.settings.promotion_min_walk_forward_pr_auc_lift,
            max_recall_drop=self.settings.promotion_max_walk_forward_recall_drop,
        )
        if completed:
            recalls = [float(row["recall"]) for row in completed]
            if float(np.std(recalls)) > 0.20:
                warnings.append("Recall instavel entre folds walk-forward.")
        warnings.extend(summary.get("failure_reasons", []))
        status = "fail" if summary.get("failure_reasons") else "warning" if warnings else "pass"
        payload = {
            "status": status,
            "warnings": warnings,
            "summary": summary,
            "folds": rows,
            "completed_fold_count": len(completed),
        }
        return self._write(output_dir, payload)

    def _write(self, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
        (output_dir / self.settings.walk_forward_report_filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        lines = [
            "# Walk-Forward Report",
            "",
            f"- Status: `{payload['status']}`",
            f"- Completed folds: {payload.get('completed_fold_count', 0)}",
            f"- Best fold: {payload.get('summary', {}).get('best_fold')}",
            f"- Worst fold: {payload.get('summary', {}).get('worst_fold')}",
            f"- Min recall fold: {payload.get('summary', {}).get('min_recall_fold')}",
            f"- Min PR-AUC fold: {payload.get('summary', {}).get('min_pr_auc_fold')}",
            f"- Last fold recall: {payload.get('summary', {}).get('last_fold_recall')}",
            f"- Last fold PR-AUC: {payload.get('summary', {}).get('last_fold_pr_auc')}",
            f"- Last fold penalty: {payload.get('summary', {}).get('last_fold_penalty')}",
            f"- Recall drop best-to-worst: {payload.get('summary', {}).get('recall_drop_best_to_worst')}",
            "",
            "## Warnings",
            "",
            *[f"- {warning}" for warning in payload.get("warnings", [])],
        ]
        (output_dir / self.settings.walk_forward_markdown_filename).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return payload

    @staticmethod
    def _fold_boundaries(row_count: int, folds: int) -> list[tuple[int, int]]:
        step = max(1, row_count // (folds + 1))
        return [
            (step * fold, min(row_count, step * (fold + 1)))
            for fold in range(1, folds + 1)
            if step * (fold + 1) <= row_count
        ]

    def _split_xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return (
            frame.drop(columns=[self.settings.target_column]),
            frame[self.settings.target_column].astype(int),
        )
