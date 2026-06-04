"""Temporal train/validation/test split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import Settings
from src.data.merge_data import first_existing


@dataclass(frozen=True)
class DataSplits:
    """Container for temporal data splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    time_column: str


class TemporalSplitter:
    """Split transactions by time without shuffling."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def split(self, df: pd.DataFrame) -> DataSplits:
        """Sort by transaction time and create train, validation and test splits."""
        time_col = first_existing(df.columns, self.settings.time_column_candidates)
        if time_col is None:
            raise ValueError("Nao foi encontrada coluna temporal para split.")

        ordered = df.copy()
        ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
        ordered = ordered.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if len(ordered) < 10:
            raise ValueError("Dataset muito pequeno para split temporal.")

        test_start = int(len(ordered) * (1 - self.settings.test_size))
        validation_start = int(len(ordered) * (1 - self.settings.test_size - self.settings.validation_size))
        train = ordered.iloc[:validation_start].copy()
        validation = ordered.iloc[validation_start:test_start].copy()
        test = ordered.iloc[test_start:].copy()

        if train.empty or validation.empty or test.empty:
            raise ValueError("Split temporal gerou particao vazia; ajuste validation_size/test_size.")

        return DataSplits(train=train, validation=validation, test=test, time_column=time_col)
