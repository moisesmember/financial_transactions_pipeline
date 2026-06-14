"""Cleaning transformers for transaction data."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import Settings
from src.data.merge_data import first_existing, normalize_columns


class FraudDataCleaner(BaseEstimator, TransformerMixin):
    """Clean raw transaction fields inside the sklearn pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudDataCleaner":
        """No-op fit to comply with sklearn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned dataframe with normalized columns and parsed values."""
        df = normalize_columns(pd.DataFrame(X))
        df = df.replace({"": np.nan, "nan": np.nan, "None": np.nan})

        amount_col = first_existing(df.columns, self.settings.amount_candidates)
        if amount_col:
            df[amount_col] = df[amount_col].map(self._parse_money)

        for column in df.columns:
            if self._looks_like_date_column(column):
                df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")

        target = self.settings.target_column
        if target in df.columns:
            df[target] = df[target].map(self._parse_binary_target).astype("int64")

        tx_id = first_existing(df.columns, ("transaction_id", "id", "trans_id"))
        if tx_id:
            df = df.drop_duplicates(subset=[tx_id], keep="first")
        return df

    @staticmethod
    def _parse_money(value: Any) -> float:
        """Parse currency-like values to float."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        text = str(value).strip()
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in {"", "-", "."}:
            return np.nan
        return float(text)

    @staticmethod
    def _parse_binary_target(value: Any) -> int:
        """Parse common binary target representations."""
        if pd.isna(value):
            return 0
        return int(str(value).strip().lower() in {"1", "true", "yes", "y", "fraud", "fraudulent"})

    @staticmethod
    def _looks_like_date_column(column: str) -> bool:
        """Identify date-like columns by name."""
        return any(token in column for token in ("date", "time", "expires", "acct_open"))
