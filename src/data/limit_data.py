"""Class-aware, time-stratified limits for local model training."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class TrainingDataLimiter:
    """Keep every positive and sample only training negatives by calendar period."""

    def __init__(
        self,
        max_rows: int | None,
        negative_positive_ratio: int = 100,
        random_state: int = 42,
    ) -> None:
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows deve ser positivo ou None.")
        if negative_positive_ratio < 1:
            raise ValueError("negative_positive_ratio deve ser pelo menos 1.")
        self.max_rows = max_rows
        self.negative_positive_ratio = negative_positive_ratio
        self.random_state = random_state

    def apply(
        self,
        training: pd.DataFrame,
        target_column: str = "is_fraud",
        time_column: str | None = None,
    ) -> pd.DataFrame:
        """Return training rows with all positives and period-stratified negatives."""
        if target_column not in training.columns:
            raise ValueError(f"Coluna target ausente no treino: {target_column}.")
        time_column = time_column or self._find_time_column(training)
        if time_column is None:
            raise ValueError("Coluna temporal obrigatoria para amostragem estratificada.")

        work = training.copy()
        work["_sampling_time"] = pd.to_datetime(work[time_column], errors="coerce")
        work["_sampling_period"] = work["_sampling_time"].dt.to_period("M")
        positives = work.loc[work[target_column].eq(1)]
        negatives = work.loc[work[target_column].eq(0)]
        if len(positives) == 0:
            raise ValueError("Treino sem positivos; amostragem segura nao pode ser aplicada.")

        sampled_negative_parts: list[pd.DataFrame] = []
        for _, period_rows in work.groupby("_sampling_period", sort=True, dropna=False):
            period_negatives = period_rows.loc[period_rows[target_column].eq(0)]
            period_positive_count = int(period_rows[target_column].eq(1).sum())
            # Keep representation for negative-only months without letting them dominate.
            period_limit = self.negative_positive_ratio * max(1, period_positive_count)
            sampled_negative_parts.append(
                self._sample(period_negatives, min(len(period_negatives), period_limit))
            )
        sampled_negatives = pd.concat(sampled_negative_parts) if sampled_negative_parts else negatives.iloc[:0]

        if self.max_rows is not None:
            negative_budget = max(0, self.max_rows - len(positives))
            if len(sampled_negatives) > negative_budget:
                sampled_negatives = self._sample_by_period(sampled_negatives, negative_budget)

        limited = pd.concat([positives, sampled_negatives]).sort_values(
            ["_sampling_time"], kind="stable"
        )
        logger.info(
            "Amostragem do treino concluida | originais=%d | positivos=%d/%d | negativos=%d/%d | ratio_max=%d | limite_preferencial=%s",
            len(training),
            int(limited[target_column].eq(1).sum()),
            len(positives),
            int(limited[target_column].eq(0).sum()),
            len(negatives),
            self.negative_positive_ratio,
            self.max_rows,
        )
        return limited.drop(columns=["_sampling_time", "_sampling_period"]).reset_index(drop=True)

    def _sample_by_period(self, frame: pd.DataFrame, budget: int) -> pd.DataFrame:
        if budget <= 0:
            return frame.iloc[:0]
        if len(frame) <= budget:
            return frame
        fraction = budget / len(frame)
        parts = []
        for _, period_rows in frame.groupby("_sampling_period", sort=True, dropna=False):
            count = min(len(period_rows), max(1, round(len(period_rows) * fraction)))
            parts.append(self._sample(period_rows, count))
        sampled = pd.concat(parts)
        if len(sampled) > budget:
            sampled = self._sample(sampled, budget)
        return sampled

    def _sample(self, frame: pd.DataFrame, count: int) -> pd.DataFrame:
        if count >= len(frame):
            return frame
        if count <= 0:
            return frame.iloc[:0]
        return frame.sample(n=count, random_state=self.random_state)

    @staticmethod
    def _find_time_column(frame: pd.DataFrame) -> str | None:
        return next(
            (
                column
                for column in frame.columns
                if any(token in column.lower() for token in ("date", "timestamp", "datetime"))
            ),
            None,
        )
