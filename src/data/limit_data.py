"""Memory-aware limits for local model training."""

from __future__ import annotations

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
        """Return at most max_rows while preserving source order."""
        if self.max_rows is None or len(transactions) <= self.max_rows:
            return transactions

        logger.info(
            "Limitando dados para treino | linhas_originais=%d | limite=%d",
            len(transactions),
            self.max_rows,
        )
        return transactions.iloc[: self.max_rows].copy()
