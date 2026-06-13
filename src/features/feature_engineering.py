"""Feature engineering with leakage-aware temporal behavior features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.settings import Settings
from src.data.merge_data import first_existing


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create temporal and behavioral features without using future rows."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.time_column_: str | None = None
        self.amount_column_: str | None = None
        self.group_column_: str | None = None
        self.history_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        """Store minimal historical rows for transform-time lag features."""
        df = pd.DataFrame(X).copy()
        self.time_column_ = first_existing(df.columns, self.settings.time_column_candidates)
        self.amount_column_ = first_existing(df.columns, self.settings.amount_candidates)
        self.group_column_ = first_existing(df.columns, self.settings.card_id_candidates) or first_existing(
            df.columns, self.settings.user_id_candidates
        )

        keep = [col for col in [self.time_column_, self.amount_column_, self.group_column_] if col]
        if keep:
            self.history_ = df[keep].copy()
        else:
            self.history_ = pd.DataFrame()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic features from current data and fitted history."""
        return self._transform(X, include_history=True)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **fit_params,
    ) -> pd.DataFrame:
        """Fit and transform training rows without prepending them as history."""
        self.fit(X, y)
        return self._transform(X, include_history=False)

    def _transform(self, X: pd.DataFrame, include_history: bool) -> pd.DataFrame:
        """Create features, optionally using fitted rows as prior history."""
        df = pd.DataFrame(X).copy()
        if self.time_column_ and self.time_column_ in df.columns:
            df[self.time_column_] = pd.to_datetime(df[self.time_column_], errors="coerce")
            df = self._add_temporal_features(df, self.time_column_)

        if self.amount_column_ and self.amount_column_ in df.columns:
            amount = pd.to_numeric(df[self.amount_column_], errors="coerce")
            df["amount_abs"] = amount.abs()
            df["amount_log1p"] = np.log1p(amount.abs())

        if self.group_column_ and self.group_column_ in df.columns and self.amount_column_ in df.columns:
            df = self._add_behavior_features(df, include_history=include_history)

        return df

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Add calendar features available at transaction time."""
        dt = df[time_col]
        df["transaction_hour"] = dt.dt.hour
        df["transaction_dayofweek"] = dt.dt.dayofweek
        df["transaction_month"] = dt.dt.month
        df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("int64")
        df["is_night"] = dt.dt.hour.between(0, 5).astype("int64")
        return df

    def _add_behavior_features(self, df: pd.DataFrame, include_history: bool = True) -> pd.DataFrame:
        """Add lagged amount statistics using only previous transactions."""
        time_col = self.time_column_
        amount_col = self.amount_column_
        group_col = self.group_column_
        if time_col is None or amount_col is None or group_col is None:
            return df

        work = df.copy()
        work["_original_order"] = np.arange(len(work))
        work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
        history = self.history_

        if include_history and history is not None and not history.empty:
            hist = history.copy()
            hist["_is_history"] = 1
            hist["_original_order"] = -1
            current = work[[time_col, amount_col, group_col, "_original_order"]].copy()
            current["_is_history"] = 0
            combined = pd.concat([hist, current], ignore_index=True, sort=False)
        else:
            combined = work[[time_col, amount_col, group_col, "_original_order"]].copy()
            combined["_is_history"] = 0

        combined[time_col] = pd.to_datetime(combined[time_col], errors="coerce")
        combined[amount_col] = pd.to_numeric(combined[amount_col], errors="coerce")
        combined = combined.sort_values([group_col, time_col, "_is_history", "_original_order"])
        grouped_amount = combined.groupby(group_col, dropna=False)[amount_col]
        previous_amount = grouped_amount.shift(1)
        combined["previous_amount"] = previous_amount
        combined["amount_mean_5_prev"] = previous_amount.groupby(combined[group_col], dropna=False).rolling(
            5, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        combined["amount_std_5_prev"] = previous_amount.groupby(combined[group_col], dropna=False).rolling(
            5, min_periods=2
        ).std().reset_index(level=0, drop=True)
        combined["transactions_seen_before"] = combined.groupby(group_col, dropna=False).cumcount()

        current_features = combined[combined["_is_history"].eq(0)].sort_values("_original_order")
        df["previous_amount"] = current_features["previous_amount"].to_numpy()
        df["amount_mean_5_prev"] = current_features["amount_mean_5_prev"].to_numpy()
        df["amount_std_5_prev"] = current_features["amount_std_5_prev"].to_numpy()
        df["transactions_seen_before"] = current_features["transactions_seen_before"].to_numpy()
        df["amount_to_mean_5_prev"] = pd.to_numeric(df[amount_col], errors="coerce") / (
            df["amount_mean_5_prev"].replace(0, np.nan)
        )
        return df
