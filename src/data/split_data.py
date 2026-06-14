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
    out_of_time: pd.DataFrame | None = None


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

        out_of_time_start = int(len(ordered) * (1 - self.settings.out_of_time_size))
        test_start = int(
            len(ordered) * (1 - self.settings.out_of_time_size - self.settings.test_size)
        )
        validation_start = int(
            len(ordered)
            * (
                1
                - self.settings.out_of_time_size
                - self.settings.test_size
                - self.settings.validation_size
            )
        )
        validation_boundary = ordered.iloc[validation_start][time_col]
        test_boundary = ordered.iloc[test_start][time_col]
        out_of_time_boundary = ordered.iloc[out_of_time_start][time_col]
        train = ordered.loc[ordered[time_col] < validation_boundary].copy()
        validation = ordered.loc[
            ordered[time_col].ge(validation_boundary) & ordered[time_col].lt(test_boundary)
        ].copy()
        test = ordered.loc[
            ordered[time_col].ge(test_boundary) & ordered[time_col].lt(out_of_time_boundary)
        ].copy()
        out_of_time = ordered.loc[ordered[time_col].ge(out_of_time_boundary)].copy()

        if train.empty or validation.empty or test.empty or out_of_time.empty:
            raise ValueError("Split temporal gerou particao vazia; ajuste validation_size/test_size.")

        return DataSplits(
            train=train,
            validation=validation,
            test=test,
            time_column=time_col,
            out_of_time=out_of_time,
        )
