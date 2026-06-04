"""Tests for data merge and temporal split."""

from __future__ import annotations

import pandas as pd

from src.config.settings import Settings
from src.data.merge_data import FraudDataMerger
from src.data.split_data import TemporalSplitter


def test_merge_adds_labels_and_mcc() -> None:
    """Transactions should be merged with labels and MCC descriptions."""
    settings = Settings()
    transactions = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2020-01-01", "2020-01-02"],
            "card_id": [10, 10],
            "client_id": [100, 100],
            "amount": ["$10.00", "$20.00"],
            "mcc": [5812, 5411],
        }
    )
    cards = pd.DataFrame({"id": [10], "client_id": [100], "card_type": ["Debit"]})
    users = pd.DataFrame({"id": [100], "current_age": [45]})
    labels = {"target": {"1": "No", "2": "Yes"}}
    mcc = {"5812": "Restaurants", "5411": "Grocery"}

    merged = FraudDataMerger(settings).merge(transactions, cards, users, mcc, labels)

    assert merged.shape[0] == 2
    assert settings.target_column in merged.columns
    assert merged[settings.target_column].tolist() == [0, 1]
    assert "mcc_description" in merged.columns


def test_temporal_split_preserves_order() -> None:
    """Temporal split should preserve chronological order."""
    settings = Settings(validation_size=0.2, test_size=0.2)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20, freq="D"),
            "is_fraud": [0, 1] * 10,
            "amount": range(20),
        }
    )

    splits = TemporalSplitter(settings).split(frame)

    assert splits.train["date"].max() < splits.validation["date"].min()
    assert splits.validation["date"].max() < splits.test["date"].min()
