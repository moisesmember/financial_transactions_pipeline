"""Tests for cleaning and feature engineering."""

from __future__ import annotations

import pandas as pd

from src.config.settings import Settings
from src.features.cleaning import FraudDataCleaner
from src.features.feature_engineering import FraudFeatureEngineer
from src.features.preprocessing import build_preprocessor, columns_to_drop


def test_cleaner_parses_money_and_target() -> None:
    """Cleaner should parse currency strings and binary fraud labels."""
    frame = pd.DataFrame({"Amount": ["$1,234.50", "$0.99"], "is_fraud": ["Yes", "No"]})

    cleaned = FraudDataCleaner(Settings()).fit_transform(frame)

    assert cleaned["amount"].tolist() == [1234.5, 0.99]
    assert cleaned["is_fraud"].tolist() == [1, 0]


def test_feature_engineering_uses_previous_amount_only() -> None:
    """Behavior features must use lagged values, not the current transaction."""
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "card_id": [1, 1, 1],
            "amount": [10.0, 20.0, 40.0],
        }
    )

    engineered = FraudFeatureEngineer(Settings()).fit_transform(frame)

    assert pd.isna(engineered.loc[0, "previous_amount"])
    assert engineered.loc[1, "previous_amount"] == 10.0
    assert engineered.loc[2, "amount_mean_5_prev"] == 15.0


def test_preprocessor_drops_raw_ids() -> None:
    """Raw identifiers should not be selected as model features."""
    settings = Settings()
    frame = pd.DataFrame(
        {
            "transaction_id": ["1"],
            "card_id": [10],
            "merchant_id": [99],
            "amount": [10.0],
            "merchant_state": ["SP"],
        }
    )

    drop_cols = columns_to_drop(frame.columns, settings)
    preprocessor = build_preprocessor(frame, settings)

    assert "transaction_id" in drop_cols
    assert "card_id" in drop_cols
    assert "merchant_id" in drop_cols
    assert preprocessor is not None
