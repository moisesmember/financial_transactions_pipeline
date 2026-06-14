"""Preprocessing builders for sklearn pipelines."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.settings import Settings


RAW_ID_HINTS = (
    "id",
    "transaction_id",
    "client_id",
    "user_id",
    "customer_id",
    "card_id",
    "card_number",
    "account_number",
    "merchant_id",
)
SENSITIVE_FEATURES = {"address", "card_number", "cvv"}
SNAPSHOT_FEATURES = {
    "card_on_dark_web",
    "credit_limit",
    "credit_score",
    "current_age",
    "num_cards_issued",
    "num_credit_cards",
    "per_capita_income",
    "retirement_age",
    "total_debt",
    "year_pin_last_changed",
    "yearly_income",
}
POST_EVENT_FEATURES = {"errors"}


def columns_to_drop(columns: Iterable[str], settings: Settings) -> list[str]:
    """Return target, raw identifiers and datetime columns that must not be model features."""
    drop: list[str] = []
    for column in columns:
        lower = column.lower()
        if lower == settings.target_column:
            drop.append(column)
        elif lower in RAW_ID_HINTS or lower.endswith("_id") or "_id_" in lower:
            drop.append(column)
        elif lower in SENSITIVE_FEATURES:
            drop.append(column)
        elif settings.strict_leakage_prevention and lower in SNAPSHOT_FEATURES | POST_EVENT_FEATURES:
            drop.append(column)
        elif any(token in lower for token in ("date", "timestamp", "datetime", "expires", "acct_open")):
            drop.append(column)
    return sorted(set(drop))


def build_preprocessor(sample: pd.DataFrame, settings: Settings) -> ColumnTransformer:
    """Build a ColumnTransformer from a transformed training sample."""
    drop_cols = columns_to_drop(sample.columns, settings)
    features = sample.drop(columns=[col for col in drop_cols if col in sample.columns], errors="ignore")
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in features.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("drop", "drop", drop_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
