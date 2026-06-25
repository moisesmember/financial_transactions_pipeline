"""Optional temporal walk-forward validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer


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
        if completed:
            recalls = [float(row["recall"]) for row in completed]
            if float(np.std(recalls)) > 0.20:
                warnings.append("Recall instavel entre folds walk-forward.")
        payload = {
            "status": "warning" if warnings else "pass",
            "warnings": warnings,
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
