"""Memory-aware limits for local model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class TrainingDataLimiter:
    """Limit raw transactions before expensive joins and transformations."""

    def __init__(self, max_rows: int | None) -> None:
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows deve ser positivo ou None.")
        self.max_rows = max_rows

    def apply(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return at most max_rows while preserving the available time horizon."""
        if self.max_rows is None or len(transactions) <= self.max_rows:
            return transactions

        date_column = next(
            (
                column
                for column in transactions.columns
                if any(token in column.lower() for token in ("date", "timestamp", "datetime"))
            ),
            None,
        )
        logger.info(
            "Limitando dados para treino | linhas_originais=%d | limite=%d | horizonte_completo=%s",
            len(transactions),
            self.max_rows,
            date_column is not None,
        )
        if date_column is not None:
            ordered = transactions.assign(
                _training_time=pd.to_datetime(transactions[date_column], errors="coerce")
            ).sort_values("_training_time", kind="stable")
            positions = np.linspace(0, len(ordered) - 1, self.max_rows, dtype=int)
            return ordered.iloc[positions].drop(columns=["_training_time"]).copy()
        return transactions.iloc[: self.max_rows].copy()
