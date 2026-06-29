"""Data drift and temporal stability reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config.settings import Settings
from src.data.split_data import DataSplits
from src.features.feature_engineering import FraudFeatureEngineer


DEFAULT_DRIFT_COLUMNS = (
    "amount",
    "amount_log1p",
    "amount_abs",
    "transaction_hour",
    "transaction_dayofweek",
    "transaction_month",
    "use_chip",
    "mcc",
    "mcc_description",
    "merchant_city",
    "merchant_state",
    "zip",
    "latitude",
    "longitude",
)


class DataDriftReportService:
    """Compare feature distributions across temporal splits."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        splits: DataSplits,
        output_dir: Path,
        important_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Write drift JSON/Markdown/CSV artifacts and return the JSON payload."""
        frames = self._engineered_split_frames(splits)
        columns = self._columns(frames, important_features)
        numeric_rows: list[dict[str, Any]] = []
        categorical_rows: list[dict[str, Any]] = []
        train = frames["train"]

        for column in columns:
            if column not in train.columns:
                continue
            if pd.api.types.is_numeric_dtype(train[column]):
                numeric_rows.extend(self._numeric_rows(column, frames))
            else:
                categorical_rows.extend(self._categorical_rows(column, frames))

        numeric = pd.DataFrame(numeric_rows)
        categorical = pd.DataFrame(categorical_rows)
        numeric_path = output_dir / self.settings.data_drift_numeric_filename
        categorical_path = output_dir / self.settings.data_drift_categorical_filename
        numeric.to_csv(numeric_path, index=False)
        categorical.to_csv(categorical_path, index=False)
        payload = self._payload(numeric, categorical)
        payload["artifacts"] = {
            "numeric": numeric_path.name,
            "categorical": categorical_path.name,
        }
        (output_dir / self.settings.data_drift_report_filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        (output_dir / self.settings.data_drift_markdown_filename).write_text(
            self._markdown(payload),
            encoding="utf-8",
        )
        return payload

    def _numeric_rows(
        self,
        column: str,
        frames: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        rows = []
        train_values = pd.to_numeric(frames["train"][column], errors="coerce").dropna()
        for split, frame in frames.items():
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            row = {
                "feature": column,
                "split": split,
                "count": int(values.size),
                "missing_rate": float(frame[column].isna().mean()),
                "mean": self._safe_float(values.mean()),
                "median": self._safe_float(values.median()),
                "std": self._safe_float(values.std()),
                "min": self._safe_float(values.min()),
                "p05": self._safe_float(values.quantile(0.05)),
                "p25": self._safe_float(values.quantile(0.25)),
                "p75": self._safe_float(values.quantile(0.75)),
                "p95": self._safe_float(values.quantile(0.95)),
                "max": self._safe_float(values.max()),
                "psi_vs_train": self._numeric_psi(train_values, values),
                "ks_statistic_vs_train": self._ks_statistic(train_values, values),
            }
            rows.append(row)
        return rows

    def _categorical_rows(
        self,
        column: str,
        frames: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        rows = []
        train_values = frames["train"][column].astype("string").fillna("<missing>")
        train_categories = set(train_values.unique())
        for split, frame in frames.items():
            values = frame[column].astype("string").fillna("<missing>")
            counts = values.value_counts(normalize=True).head(10)
            categories = set(values.unique())
            rows.append(
                {
                    "feature": column,
                    "split": split,
                    "count": int(len(values)),
                    "missing_rate": float(frame[column].isna().mean()),
                    "unique_count": int(values.nunique()),
                    "new_category_count_vs_train": int(len(categories - train_categories)),
                    "missing_category_count_vs_train": int(len(train_categories - categories)),
                    "psi_vs_train": self._categorical_psi(train_values, values),
                    "top_categories": json.dumps(counts.to_dict(), ensure_ascii=True),
                }
            )
        return rows

    def _payload(self, numeric: pd.DataFrame, categorical: pd.DataFrame) -> dict[str, Any]:
        warnings: list[str] = []
        if not numeric.empty:
            high_numeric = numeric.loc[numeric["psi_vs_train"].fillna(0).ge(0.25)]
            warnings.extend(
                f"Numeric drift alto em {row.feature}/{row.split}: PSI={row.psi_vs_train:.3f}"
                for row in high_numeric.itertuples()
                if row.split != "train"
            )
        if not categorical.empty:
            high_categorical = categorical.loc[categorical["psi_vs_train"].fillna(0).ge(0.25)]
            warnings.extend(
                f"Categorical drift alto em {row.feature}/{row.split}: PSI={row.psi_vs_train:.3f}"
                for row in high_categorical.itertuples()
                if row.split != "train"
            )
        return {
            "status": "warning" if warnings else "pass",
            "warnings": warnings,
            "numeric_feature_count": int(numeric["feature"].nunique()) if not numeric.empty else 0,
            "categorical_feature_count": int(categorical["feature"].nunique()) if not categorical.empty else 0,
            "max_numeric_psi": self._safe_float(numeric["psi_vs_train"].max()) if not numeric.empty else None,
            "max_categorical_psi": self._safe_float(categorical["psi_vs_train"].max()) if not categorical.empty else None,
        }

    def _columns(
        self,
        frames: dict[str, pd.DataFrame],
        important_features: list[str] | None,
    ) -> list[str]:
        available = set().union(*(set(frame.columns) for frame in frames.values()))
        candidates = list(DEFAULT_DRIFT_COLUMNS)
        if important_features:
            candidates.extend(important_features)
        seen: set[str] = set()
        return [
            column
            for column in candidates
            if column in available and not (column in seen or seen.add(column))
        ]

    @staticmethod
    def _split_frames(splits: DataSplits) -> dict[str, pd.DataFrame]:
        frames = {
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        }
        if splits.out_of_time is not None:
            frames["out_of_time"] = splits.out_of_time
        return frames

    def _engineered_split_frames(self, splits: DataSplits) -> dict[str, pd.DataFrame]:
        """Apply the same deterministic feature engineering used by training."""
        raw_frames = self._split_frames(splits)
        engineer = FraudFeatureEngineer(self.settings)
        frames = {"train": engineer.fit_transform(raw_frames["train"])}
        for split, frame in raw_frames.items():
            if split == "train":
                continue
            frames[split] = engineer.transform(frame)
        return frames

    @staticmethod
    def _numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float | None:
        if expected.empty or actual.empty or expected.nunique() < 2:
            return None
        quantiles = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            return None
        expected_counts, _ = np.histogram(expected, bins=quantiles)
        actual_counts, _ = np.histogram(actual, bins=quantiles)
        return DataDriftReportService._psi_from_counts(expected_counts, actual_counts)

    @staticmethod
    def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float | None:
        categories = sorted(set(expected.unique()) | set(actual.unique()))
        if not categories:
            return None
        expected_counts = expected.value_counts().reindex(categories, fill_value=0).to_numpy()
        actual_counts = actual.value_counts().reindex(categories, fill_value=0).to_numpy()
        return DataDriftReportService._psi_from_counts(expected_counts, actual_counts)

    @staticmethod
    def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
        epsilon = 1e-6
        expected = expected_counts / max(1, expected_counts.sum())
        actual = actual_counts / max(1, actual_counts.sum())
        expected = np.clip(expected, epsilon, None)
        actual = np.clip(actual, epsilon, None)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    @staticmethod
    def _ks_statistic(expected: pd.Series, actual: pd.Series) -> float | None:
        if expected.empty or actual.empty:
            return None
        return float(ks_2samp(expected, actual).statistic)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    @staticmethod
    def _markdown(payload: dict[str, Any]) -> str:
        lines = [
            "# Data Drift Report",
            "",
            f"- Status: `{payload['status']}`",
            f"- Numeric features: {payload['numeric_feature_count']}",
            f"- Categorical features: {payload['categorical_feature_count']}",
            f"- Max numeric PSI: {payload['max_numeric_psi']}",
            f"- Max categorical PSI: {payload['max_categorical_psi']}",
            "",
            "## Warnings",
            "",
            *[f"- {warning}" for warning in payload.get("warnings", [])],
        ]
        return "\n".join(lines) + "\n"
