"""Tests for data merge and temporal split."""

from __future__ import annotations

import pandas as pd

from src.config.settings import Settings
from src.data.limit_data import TrainingDataLimiter
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
    assert splits.out_of_time is not None
    assert splits.test["date"].max() < splits.out_of_time["date"].min()


def test_temporal_split_keeps_equal_timestamps_in_same_partition() -> None:
    settings = Settings(validation_size=0.2, test_size=0.2)
    dates = list(pd.date_range("2020-01-01", periods=18, freq="D"))
    dates[12] = dates[11]
    frame = pd.DataFrame(
        {
            "date": dates,
            "is_fraud": [0, 1] * 9,
            "amount": range(18),
        }
    )

    splits = TemporalSplitter(settings).split(frame)

    assert splits.train["date"].max() < splits.validation["date"].min()
    assert splits.validation["date"].max() < splits.test["date"].min()


def test_training_data_limiter_preserves_source_order() -> None:
    """Training row limits should be applied before expensive processing."""
    transactions = pd.DataFrame({"id": range(10), "amount": range(10)})

    limited = TrainingDataLimiter(max_rows=4).apply(transactions)

    assert limited["id"].tolist() == [0, 1, 2, 3]
    assert len(transactions) == 10


def test_training_data_limiter_preserves_full_time_horizon() -> None:
    transactions = pd.DataFrame(
        {
            "id": range(10),
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        }
    )

    limited = TrainingDataLimiter(max_rows=4).apply(transactions)

    assert limited["date"].min() == transactions["date"].min()
    assert limited["date"].max() == transactions["date"].max()
