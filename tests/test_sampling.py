"""Regression tests for safe temporal negative sampling and its audit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config.settings import Settings
from src.data.limit_data import temporal_stratified_negative_sampling
from src.data.split_data import DataSplits
from src.models.sampling_audit import SamplingAuditService


def _training_frame() -> pd.DataFrame:
    rows = []
    row_id = 0
    for month in range(1, 7):
        for offset in range(10):
            rows.append(
                {
                    "id": row_id,
                    "date": pd.Timestamp(2020, month, 1) + pd.Timedelta(days=offset),
                    "is_fraud": 1 if offset == 0 else 0,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_temporal_sampling_preserves_positives_and_multiple_periods() -> None:
    training = _training_frame()

    sampled = temporal_stratified_negative_sampling(
        training,
        target_col="is_fraud",
        date_col="date",
        max_rows=24,
        negative_to_positive_ratio=100,
        period="M",
        random_state=7,
    )

    positive_ids = set(training.loc[training["is_fraud"].eq(1), "id"])
    assert positive_ids <= set(sampled["id"])
    assert len(sampled) == 24
    assert sampled["date"].is_monotonic_increasing
    assert sampled["date"].dt.to_period("M").nunique() == 6


def test_temporal_sampling_is_reproducible_and_reduces_only_negatives() -> None:
    training = _training_frame()
    kwargs = {
        "target_col": "is_fraud",
        "date_col": "date",
        "max_rows": None,
        "negative_to_positive_ratio": 2,
        "period": "M",
        "random_state": 19,
    }

    first = temporal_stratified_negative_sampling(training, **kwargs)
    second = temporal_stratified_negative_sampling(training, **kwargs)

    assert first["id"].tolist() == second["id"].tolist()
    assert int(first["is_fraud"].sum()) == int(training["is_fraud"].sum())
    assert int(first["is_fraud"].eq(0).sum()) == 12


def test_temporal_sampling_fails_instead_of_removing_positives() -> None:
    training = _training_frame()

    with pytest.raises(ValueError, match="nao removera positivos"):
        temporal_stratified_negative_sampling(
            training,
            target_col="is_fraud",
            date_col="date",
            max_rows=5,
        )


def test_sampling_audit_generates_all_artifacts_and_keeps_evaluation_full(tmp_path) -> None:
    frame = _training_frame()
    train_before = frame.iloc[:30].copy()
    train_after = temporal_stratified_negative_sampling(
        train_before,
        target_col="is_fraud",
        date_col="date",
        max_rows=15,
        negative_to_positive_ratio=100,
    )
    validation = frame.iloc[30:40].copy()
    test = frame.iloc[40:50].copy()
    oot = frame.iloc[50:].copy()
    before = DataSplits(train_before, validation, test, "date", oot)
    after = DataSplits(train_after, validation, test, "date", oot)
    settings = Settings(
        project_root=tmp_path,
        raw_data_max_rows=0,
        training_max_rows=15,
        negative_sampling_by="month",
    )

    report = SamplingAuditService(settings).build(
        raw_transactions=frame.drop(columns="is_fraud"),
        supervised=frame,
        splits_before_sampling=before,
        splits_after_sampling=after,
        target_audit={"missing_transaction_labels_count": 0},
        output_dir=tmp_path,
    )

    assert report["positives_preserved_pct"] == 1.0
    assert report["validation_test_oot_limited"] is False
    assert report["training_limit_applied_stage"].startswith("after_supervised")
    for filename in (
        settings.sampling_audit_filename,
        settings.sampling_audit_markdown_filename,
        settings.sampling_by_period_filename,
        settings.sampling_by_split_filename,
        settings.sampling_positive_coverage_filename,
    ):
        assert (tmp_path / filename).exists()
    payload = json.loads((tmp_path / settings.sampling_audit_filename).read_text())
    assert payload["positives_removed_by_sampling"] == 0


def test_sampling_audit_writes_failure_then_blocks_positive_loss(tmp_path) -> None:
    frame = _training_frame()
    train_before = frame.iloc[:30].copy()
    train_after = train_before.loc[
        ~train_before["id"].eq(train_before.loc[train_before["is_fraud"].eq(1), "id"].iloc[0])
    ].copy()
    validation = frame.iloc[30:40].copy()
    test = frame.iloc[40:50].copy()
    oot = frame.iloc[50:].copy()
    before = DataSplits(train_before, validation, test, "date", oot)
    after = DataSplits(train_after, validation, test, "date", oot)
    settings = Settings(project_root=tmp_path, training_max_rows=20)

    with pytest.raises(RuntimeError, match="positivos conhecidos"):
        SamplingAuditService(settings).build(
            raw_transactions=frame.drop(columns="is_fraud"),
            supervised=frame,
            splits_before_sampling=before,
            splits_after_sampling=after,
            target_audit={"missing_transaction_labels_count": 0},
            output_dir=tmp_path,
        )

    payload = json.loads((tmp_path / settings.sampling_audit_filename).read_text())
    assert payload["status"] == "fail"
    assert payload["positives_preserved_pct"] < 1.0
