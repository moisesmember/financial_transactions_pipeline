"""Class-aware, time-stratified limits for local model training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


def temporal_stratified_negative_sampling(
    train_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    max_rows: int | None,
    negative_to_positive_ratio: int | None = None,
    period: str = "M",
    random_state: int = 42,
    enforce_fixed_ratio: bool = False,
) -> pd.DataFrame:
    """Preserve all positives and sample only negatives across temporal strata.

    ``max_rows=0`` is the explicit full-dataset mode. ``None`` means that only
    the optional negative/positive ratio can reduce the training set.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Coluna target ausente no treino: {target_col}.")
    if date_col not in train_df.columns:
        raise ValueError(f"Coluna temporal ausente no treino: {date_col}.")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows deve ser zero, positivo ou None.")
    if negative_to_positive_ratio is not None and negative_to_positive_ratio < 1:
        raise ValueError("negative_to_positive_ratio deve ser pelo menos 1 ou None.")
    normalized_period = period.strip().upper()
    if normalized_period not in {"M", "Y"}:
        raise ValueError("period deve ser 'M' (mes) ou 'Y' (ano).")

    target = pd.to_numeric(train_df[target_col], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("O target do treino deve conter apenas labels validos 0 e 1.")
    sampling_time = pd.to_datetime(train_df[date_col], errors="coerce")
    if sampling_time.isna().any():
        raise ValueError("Datas invalidas nao sao permitidas na amostragem temporal.")

    # Explicit unlimited mode: no row is sampled, even when a ratio is configured.
    if max_rows == 0:
        return train_df.assign(_sampling_time=sampling_time).sort_values(
            "_sampling_time", kind="stable"
        ).drop(columns="_sampling_time").reset_index(drop=True)

    work = train_df.copy()
    work["_sampling_time"] = sampling_time
    work["_sampling_period"] = sampling_time.dt.to_period(normalized_period)
    positives = work.loc[target.eq(1)]
    negatives = work.loc[target.eq(0)]
    positive_count = len(positives)
    if positive_count == 0:
        raise ValueError("Treino sem positivos; amostragem segura nao pode ser aplicada.")
    if max_rows is not None and positive_count > max_rows:
        raise ValueError(
            "TRAINING_MAX_ROWS e menor que o numero de positivos conhecidos; "
            "a pipeline nao removera positivos. Aumente o limite ou use 0."
        )

    negative_budget = len(negatives)
    if negative_to_positive_ratio is not None:
        negative_budget = min(
            negative_budget,
            positive_count * negative_to_positive_ratio,
        )
    fixed_ratio_feasible = bool(
        enforce_fixed_ratio
        and negative_to_positive_ratio is not None
        and len(negatives) >= positive_count * negative_to_positive_ratio
    )
    if max_rows is not None and not fixed_ratio_feasible:
        negative_budget = min(negative_budget, max_rows - positive_count)
    elif (
        max_rows is not None
        and positive_count + negative_budget > max_rows
    ):
        logger.warning(
            "TRAINING_MAX_ROWS excedido para preservar a razao fixa | limite=%d | requerido=%d",
            max_rows,
            positive_count + negative_budget,
        )

    sampled_negatives = _sample_negatives_by_period(
        negatives,
        budget=negative_budget,
        random_state=random_state,
    )
    sampled = pd.concat([positives, sampled_negatives], ignore_index=False).sort_values(
        ["_sampling_time"], kind="stable"
    )
    sampled_positive_count = int(sampled[target_col].eq(1).sum())
    if sampled_positive_count != positive_count:
        raise RuntimeError(
            "Falha critica: a amostragem removeu positivos conhecidos do treino."
        )
    logger.info(
        "Amostragem temporal do treino | antes=%d | depois=%d | positivos=%d/%d | negativos=%d/%d | periodo=%s | max_rows=%s | ratio=%s",
        len(train_df),
        len(sampled),
        sampled_positive_count,
        positive_count,
        int(sampled[target_col].eq(0).sum()),
        len(negatives),
        normalized_period,
        max_rows,
        negative_to_positive_ratio,
    )
    return sampled.drop(columns=["_sampling_time", "_sampling_period"]).reset_index(drop=True)


def _sample_negatives_by_period(
    negatives: pd.DataFrame,
    budget: int,
    random_state: int,
) -> pd.DataFrame:
    """Allocate an exact negative budget proportionally over time periods."""
    if budget <= 0:
        return negatives.iloc[:0]
    if budget >= len(negatives):
        return negatives

    counts = negatives.groupby("_sampling_period", sort=True, dropna=False).size()
    period_count = len(counts)
    allocation = pd.Series(0, index=counts.index, dtype="int64")

    if budget < period_count:
        # When one row per period is impossible, spread represented periods over
        # the complete time range instead of selecting only the earliest periods.
        positions = np.linspace(0, period_count - 1, num=budget, dtype=int)
        allocation.iloc[np.unique(positions)] = 1
    else:
        allocation[:] = 1
        remaining = budget - period_count
        capacity = counts - allocation
        if remaining > 0 and int(capacity.sum()) > 0:
            quotas = remaining * capacity / capacity.sum()
            extra = np.floor(quotas).astype(int)
            allocation += extra
            remainder = remaining - int(extra.sum())
            if remainder:
                order = (quotas - extra).sort_values(ascending=False, kind="stable").index
                for period_key in order:
                    if remainder == 0:
                        break
                    if allocation.loc[period_key] < counts.loc[period_key]:
                        allocation.loc[period_key] += 1
                        remainder -= 1

    parts: list[pd.DataFrame] = []
    for period_key, rows in negatives.groupby("_sampling_period", sort=True, dropna=False):
        count = int(allocation.loc[period_key])
        if count <= 0:
            continue
        if count >= len(rows):
            parts.append(rows)
        else:
            # A period-specific seed makes the result stable if group iteration changes.
            seed = (random_state + sum(ord(char) for char in str(period_key))) % (2**32 - 1)
            parts.append(rows.sample(n=count, random_state=seed))
    if not parts:
        return negatives.iloc[:0]
    sampled = pd.concat(parts)
    if len(sampled) != budget:
        raise RuntimeError(
            f"Falha ao distribuir o orcamento de negativos: esperado={budget}, obtido={len(sampled)}."
        )
    return sampled


class TrainingDataLimiter:
    """Compatibility wrapper for safe temporal negative sampling."""

    def __init__(
        self,
        max_rows: int | None,
        negative_positive_ratio: int | None = 100,
        random_state: int = 42,
        period: str = "M",
        enforce_fixed_ratio: bool = False,
    ) -> None:
        self.max_rows = max_rows
        self.negative_positive_ratio = negative_positive_ratio
        self.random_state = random_state
        self.period = period
        self.enforce_fixed_ratio = enforce_fixed_ratio

    def apply(
        self,
        training: pd.DataFrame,
        target_column: str = "is_fraud",
        time_column: str | None = None,
    ) -> pd.DataFrame:
        """Return training rows with all positives and stratified negatives."""
        time_column = time_column or self._find_time_column(training)
        if time_column is None:
            raise ValueError("Coluna temporal obrigatoria para amostragem estratificada.")
        return temporal_stratified_negative_sampling(
            training,
            target_col=target_column,
            date_col=time_column,
            max_rows=self.max_rows,
            negative_to_positive_ratio=self.negative_positive_ratio,
            period=self.period,
            random_state=self.random_state,
            enforce_fixed_ratio=self.enforce_fixed_ratio,
        )

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
