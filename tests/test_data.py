"""Tests for data merge and temporal split."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.limit_data import TrainingDataLimiter
from src.data.load_data import RawDataRepository
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


def test_raw_data_max_rows_is_separate_from_training_limit(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "transactions_data.csv").write_text(
        "id,date,amount\n1,2020-01-01,10\n2,2020-01-02,20\n3,2020-01-03,30\n",
        encoding="utf-8",
    )
    settings = Settings(
        project_root=tmp_path,
        raw_data_dir=raw_dir,
        storage_backend="local",
        keep_local_raw_data=True,
        keep_local_artifacts=True,
        raw_data_max_rows=2,
        training_max_rows=1,
    )

    loaded = RawDataRepository(settings).load_transactions()

    assert loaded["id"].tolist() == [1, 2]


def test_merge_uses_inner_join_for_unlabeled_transactions() -> None:
    settings = Settings()
    transactions = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    merged = FraudDataMerger(settings).merge(
        transactions,
        pd.DataFrame(),
        pd.DataFrame(),
        {},
        {"target": {"1": "No", "2": "Yes"}},
    )

    assert merged["transaction_id"].tolist() == ["1", "2"]
    assert merged[settings.target_column].tolist() == [0, 1]


@pytest.mark.parametrize("bad_label", [None, "", float("nan"), "unknown"])
def test_merge_rejects_unknown_label_values(bad_label) -> None:
    settings = Settings()
    transactions = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2020-01-01", "2020-01-02"],
            "amount": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="nao podem ser tratados como classe 0"):
        FraudDataMerger(settings).merge(
            transactions,
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            {"target": {"1": "No", "2": bad_label}},
        )


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


def test_training_data_limiter_preserves_every_positive_and_only_samples_negatives() -> None:
    training = pd.DataFrame(
        {
            "id": range(12),
            "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
            "is_fraud": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        }
    )

    limited = TrainingDataLimiter(max_rows=7, negative_positive_ratio=2).apply(training)

    assert set(training.loc[training["is_fraud"].eq(1), "id"]) <= set(limited["id"])
    assert int(limited["is_fraud"].sum()) == 4
    assert len(limited) == 7


def test_training_data_limiter_distributes_ratio_budget_across_months() -> None:
    dates = [pd.Timestamp("2020-01-01")] * 12 + [pd.Timestamp("2020-02-01")] * 8
    training = pd.DataFrame(
        {
            "id": range(20),
            "date": dates,
            "is_fraud": [1, 1] + [0] * 10 + [1] + [0] * 7,
        }
    )

    limited = TrainingDataLimiter(max_rows=None, negative_positive_ratio=2).apply(training)
    monthly = limited.assign(month=limited["date"].dt.to_period("M")).groupby("month")["is_fraud"]

    assert monthly.sum().to_dict() == {
        pd.Period("2020-01", freq="M"): 2,
        pd.Period("2020-02", freq="M"): 1,
    }
    monthly_sizes = limited.groupby(limited["date"].dt.to_period("M")).size()
    assert int(limited["is_fraud"].eq(0).sum()) == 6
    assert set(monthly_sizes.index) == {
        pd.Period("2020-01", freq="M"),
        pd.Period("2020-02", freq="M"),
    }


def test_training_data_limiter_rejects_limit_below_positive_count() -> None:
    training = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "is_fraud": [1, 1, 1, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="menor que o numero de positivos"):
        TrainingDataLimiter(max_rows=2).apply(training)


def test_training_data_limiter_zero_returns_complete_supervised_training() -> None:
    training = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "is_fraud": [1] + [0] * 9,
        }
    )

    complete = TrainingDataLimiter(max_rows=0, negative_positive_ratio=1).apply(training)

    assert len(complete) == len(training)
    assert int(complete["is_fraud"].sum()) == 1
