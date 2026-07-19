"""Sampling integrity audit and human-readable artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings
from src.data.merge_data import first_existing, normalize_columns
from src.data.split_data import DataSplits
from src.utils.logger import get_logger


logger = get_logger(__name__)


class SamplingAuditService:
    """Prove that training sampling happened after split and kept every positive."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        raw_transactions: pd.DataFrame,
        supervised: pd.DataFrame,
        splits_before_sampling: DataSplits,
        splits_after_sampling: DataSplits,
        target_audit: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Write JSON, Markdown and CSV audit artifacts and return the payload."""
        target_col = self.settings.target_column
        time_col = splits_before_sampling.time_column
        train_before = splits_before_sampling.train
        train_after = splits_after_sampling.train
        positives_before = int(train_before[target_col].eq(1).sum())
        positives_after = int(train_after[target_col].eq(1).sum())
        negatives_before = int(train_before[target_col].eq(0).sum())
        negatives_after = int(train_after[target_col].eq(0).sum())
        effective_negative_ratio = (
            negatives_after / positives_after if positives_after else None
        )
        configured_negative_ratio = self.settings.training_negative_positive_ratio
        sampling_ratio_deviation = (
            effective_negative_ratio - configured_negative_ratio
            if effective_negative_ratio is not None and configured_negative_ratio is not None
            else None
        )
        positives_removed = positives_before - positives_after
        positive_coverage = (
            positives_after / positives_before if positives_before else 1.0
        )
        before_period_bounds = self._period_bounds(train_before, time_col)
        after_period_bounds = self._period_bounds(train_after, time_col)
        temporal_range_preserved = before_period_bounds == after_period_bounds
        evaluation_splits_unchanged = (
            len(splits_before_sampling.validation) == len(splits_after_sampling.validation)
            and len(splits_before_sampling.test) == len(splits_after_sampling.test)
            and (
                splits_before_sampling.out_of_time is None
                and splits_after_sampling.out_of_time is None
                or splits_before_sampling.out_of_time is not None
                and splits_after_sampling.out_of_time is not None
                and len(splits_before_sampling.out_of_time)
                == len(splits_after_sampling.out_of_time)
            )
        )

        normalized_raw_transactions = normalize_columns(raw_transactions)
        full_time_col = first_existing(
            normalized_raw_transactions.columns,
            self.settings.time_column_candidates,
        )
        full_min, full_max = self._time_bounds(normalized_raw_transactions, full_time_col)
        supervised_min, supervised_max = self._time_bounds(supervised, time_col)
        supervised_positives = int(supervised[target_col].eq(1).sum())

        by_split = self._by_split(
            normalized_raw_transactions,
            supervised,
            splits_before_sampling,
            splits_after_sampling,
            full_time_col,
        )
        by_period = self._by_period(
            supervised,
            splits_before_sampling,
            splits_after_sampling,
        )
        positive_coverage_frame = self._positive_coverage(
            splits_before_sampling,
            splits_after_sampling,
        )

        warnings: list[str] = []
        failures: list[str] = []
        if positives_removed != 0 or positive_coverage < 1.0:
            failures.append(
                "Falha critica: positivos conhecidos foram removidos pela amostragem."
            )
        if not temporal_range_preserved:
            warnings.append(
                "WARNING CRITICO: a amostragem encurtou o range de periodos do treino."
            )
        if not evaluation_splits_unchanged:
            failures.append(
                "Falha critica: validation, test ou out_of_time foram reduzidos pela amostragem."
            )
        if self.settings.raw_data_max_rows:
            warnings.append(
                "RAW_DATA_MAX_ROWS esta ativo; a rodada nao representa o dataset bruto completo."
            )

        payload: dict[str, Any] = {
            "status": "fail" if failures else ("warning" if warnings else "pass"),
            "sampling_reliable": (
                not failures
                and temporal_range_preserved
                and not self.settings.raw_data_max_rows
            ),
            "raw_data_max_rows": self.settings.raw_data_max_rows,
            "training_max_rows": self.settings.training_max_rows,
            "preserve_all_positives": self.settings.preserve_all_positives,
            "negative_sampling_enabled": self.settings.negative_sampling_enabled,
            "negative_sampling_strategy": self.settings.negative_sampling_strategy,
            "negative_sampling_by": self.settings.negative_sampling_by,
            "negative_to_positive_ratio": self.settings.training_negative_positive_ratio,
            "configured_negative_ratio": configured_negative_ratio,
            "effective_negative_ratio": effective_negative_ratio,
            "sampling_ratio_deviation": sampling_ratio_deviation,
            "sampling_enforce_fixed_ratio": self.settings.sampling_enforce_fixed_ratio,
            "positive_rate_before_sampling": (
                positives_before / (positives_before + negatives_before)
                if positives_before + negatives_before
                else None
            ),
            "positive_rate_after_sampling": (
                positives_after / (positives_after + negatives_after)
                if positives_after + negatives_after
                else None
            ),
            "training_limit_applied": len(train_after) < len(train_before),
            "training_limit_applied_stage": "after_supervised_inner_join_and_temporal_split_train_only",
            "sequential_training_limit_detected": False,
            "validation_test_oot_limited": not evaluation_splits_unchanged,
            "full_dataset": {
                "rows": int(len(raw_transactions)),
                "min_date": full_min,
                "max_date": full_max,
                "positive_count": supervised_positives,
                "fraud_rate": (
                    supervised_positives / len(raw_transactions)
                    if len(raw_transactions)
                    else 0.0
                ),
                "is_raw_load_limited": bool(self.settings.raw_data_max_rows),
            },
            "supervised_dataset": {
                "rows": int(len(supervised)),
                "min_date": supervised_min,
                "max_date": supervised_max,
                "positive_count": supervised_positives,
                "fraud_rate": float(supervised[target_col].mean()) if len(supervised) else 0.0,
                "transactions_without_label_removed": int(
                    target_audit.get("missing_transaction_labels_count", 0)
                ),
                "join_type": "inner",
                "unknown_labels_used_as_negative": False,
            },
            "training_before_sampling": self._summary(train_before, time_col),
            "training_after_sampling": self._summary(train_after, time_col),
            "positives_removed_by_sampling": positives_removed,
            "positives_preserved_pct": positive_coverage,
            "temporal_period_range_before": list(before_period_bounds),
            "temporal_period_range_after": list(after_period_bounds),
            "temporal_range_preserved": temporal_range_preserved,
            "warnings": warnings,
            "failures": failures,
            "by_split": by_split.to_dict(orient="records"),
            "positive_coverage_by_split": positive_coverage_frame.to_dict(orient="records"),
            "artifacts": {
                "by_period": self.settings.sampling_by_period_filename,
                "by_split": self.settings.sampling_by_split_filename,
                "positive_coverage": self.settings.sampling_positive_coverage_filename,
            },
        }

        payload = self._json_safe(payload)
        output_dir.mkdir(parents=True, exist_ok=True)
        by_period.to_csv(output_dir / self.settings.sampling_by_period_filename, index=False)
        by_split.to_csv(output_dir / self.settings.sampling_by_split_filename, index=False)
        positive_coverage_frame.to_csv(
            output_dir / self.settings.sampling_positive_coverage_filename,
            index=False,
        )
        (output_dir / self.settings.sampling_audit_filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        (output_dir / self.settings.sampling_audit_markdown_filename).write_text(
            self._markdown(payload),
            encoding="utf-8",
        )
        if failures:
            raise RuntimeError("; ".join(failures))
        if warnings:
            logger.warning("Sampling audit: %s", " ".join(warnings))
        return payload

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _by_split(
        self,
        raw_transactions: pd.DataFrame,
        supervised: pd.DataFrame,
        before: DataSplits,
        after: DataSplits,
        raw_time_col: str | None,
    ) -> pd.DataFrame:
        frames: list[tuple[str, pd.DataFrame, str | None]] = [
            ("dataset_full", raw_transactions, raw_time_col),
            ("dataset_supervised", supervised, before.time_column),
            ("train_before_sampling", before.train, before.time_column),
            ("train_after_sampling", after.train, after.time_column),
            ("validation", before.validation, before.time_column),
            ("test", before.test, before.time_column),
        ]
        if before.out_of_time is not None:
            frames.append(("out_of_time", before.out_of_time, before.time_column))
        rows = []
        for name, frame, date_col in frames:
            summary = self._summary(frame, date_col)
            rows.append({"split": name, **summary})
        return pd.DataFrame(rows)

    def _by_period(
        self,
        supervised: pd.DataFrame,
        before: DataSplits,
        after: DataSplits,
    ) -> pd.DataFrame:
        frames: list[tuple[str, pd.DataFrame]] = [
            ("dataset_supervised", supervised),
            ("train_before_sampling", before.train),
            ("train_after_sampling", after.train),
            ("validation", before.validation),
            ("test", before.test),
        ]
        if before.out_of_time is not None:
            frames.append(("out_of_time", before.out_of_time))
        rows: list[dict[str, Any]] = []
        for granularity, freq in (("month", "M"), ("year", "Y")):
            for split_name, frame in frames:
                work = frame[[before.time_column, self.settings.target_column]].copy()
                work["period"] = (
                    pd.to_datetime(work[before.time_column]).dt.to_period(freq).astype(str)
                )
                for period, group in work.groupby("period", sort=True, dropna=False):
                    positives = int(group[self.settings.target_column].eq(1).sum())
                    negatives = int(group[self.settings.target_column].eq(0).sum())
                    rows.append(
                        {
                            "split": split_name,
                            "granularity": granularity,
                            "period": period,
                            "rows": int(len(group)),
                            "positive_count": positives,
                            "negative_count": negatives,
                            "fraud_rate": positives / len(group) if len(group) else 0.0,
                            "effective_negative_ratio": (
                                negatives / positives if positives else None
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def _positive_coverage(self, before: DataSplits, after: DataSplits) -> pd.DataFrame:
        before_frames = self._split_frames(before)
        after_frames = self._split_frames(after)
        rows = []
        for name, frame_before in before_frames.items():
            frame_after = after_frames[name]
            positive_before = int(frame_before[self.settings.target_column].eq(1).sum())
            positive_after = int(frame_after[self.settings.target_column].eq(1).sum())
            rows.append(
                {
                    "split": name,
                    "positive_count_before": positive_before,
                    "positive_count_after": positive_after,
                    "positives_removed": positive_before - positive_after,
                    "positives_preserved_pct": (
                        positive_after / positive_before if positive_before else 1.0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _summary(self, frame: pd.DataFrame, date_col: str | None) -> dict[str, Any]:
        target_col = self.settings.target_column
        positives = int(frame[target_col].eq(1).sum()) if target_col in frame else None
        negatives = int(frame[target_col].eq(0).sum()) if target_col in frame else None
        date_min, date_max = self._time_bounds(frame, date_col)
        return {
            "rows": int(len(frame)),
            "positive_count": positives,
            "negative_count": negatives,
            "fraud_rate": (
                positives / len(frame) if positives is not None and len(frame) else None
            ),
            "min_date": date_min,
            "max_date": date_max,
        }

    @staticmethod
    def _split_frames(splits: DataSplits) -> dict[str, pd.DataFrame]:
        frames = {
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        }
        if splits.out_of_time is not None:
            frames["out_of_time"] = splits.out_of_time
        return frames

    @staticmethod
    def _time_bounds(frame: pd.DataFrame, date_col: str | None) -> tuple[str | None, str | None]:
        if date_col is None or date_col not in frame or frame.empty:
            return None, None
        dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
        if dates.empty:
            return None, None
        return dates.min().isoformat(), dates.max().isoformat()

    def _period_bounds(self, frame: pd.DataFrame, date_col: str) -> tuple[str | None, str | None]:
        freq = "M" if self.settings.negative_sampling_by == "month" else "Y"
        periods = pd.to_datetime(frame[date_col], errors="coerce").dropna().dt.to_period(freq)
        if periods.empty:
            return None, None
        return str(periods.min()), str(periods.max())

    @staticmethod
    def _markdown(payload: dict[str, Any]) -> str:
        before = payload["training_before_sampling"]
        after = payload["training_after_sampling"]
        trust = "confiavel" if payload["sampling_reliable"] else "nao confiavel"
        lines = [
            "# Sampling Audit",
            "",
            f"- Status: `{payload['status']}`",
            f"- Rodada: **{trust}** para avaliacao temporal.",
            f"- Limite bruto: `{payload['raw_data_max_rows']}`",
            f"- Limite de treino: `{payload['training_max_rows']}`",
            f"- Estagio do limite: `{payload['training_limit_applied_stage']}`",
            f"- Estrategia: `{payload['negative_sampling_strategy']}` por `{payload['negative_sampling_by']}`",
            f"- Ratio negativo/positivo: `{payload['negative_to_positive_ratio']}`",
            f"- Ratio efetivo: `{payload['effective_negative_ratio']}`",
            f"- Desvio da ratio configurada: `{payload['sampling_ratio_deviation']}`",
            f"- Prevalencia antes/depois: `{payload['positive_rate_before_sampling']}` / `{payload['positive_rate_after_sampling']}`",
            f"- Positivos antes/depois: {before['positive_count']} / {after['positive_count']}",
            f"- Positivos preservados: {payload['positives_preserved_pct']:.2%}",
            f"- Negativos antes/depois: {before['negative_count']} / {after['negative_count']}",
            f"- Linhas antes/depois: {before['rows']} / {after['rows']}",
            f"- Range temporal antes: {before['min_date']} a {before['max_date']}",
            f"- Range temporal depois: {after['min_date']} a {after['max_date']}",
            f"- Range de periodos preservado: `{payload['temporal_range_preserved']}`",
            "",
            "## Warnings",
            "",
            *[f"- {item}" for item in payload.get("warnings", [])],
            "",
            "## Failures",
            "",
            *[f"- {item}" for item in payload.get("failures", [])],
            "",
            "## Split Comparison",
            "",
            "| Split | Rows | Positives | Negatives | Fraud rate | Min date | Max date |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for row in payload["by_split"]:
            rate = "n/a" if row["fraud_rate"] is None else f"{row['fraud_rate']:.6f}"
            lines.append(
                f"| {row['split']} | {row['rows']} | {row['positive_count']} | "
                f"{row['negative_count']} | {rate} | {row['min_date']} | {row['max_date']} |"
            )
        if payload["sequential_training_limit_detected"]:
            lines.extend(
                [
                    "",
                    "A rodada nao e confiavel para avaliacao temporal, pois o limite de linhas produziu vies temporal.",
                ]
            )
        return "\n".join(lines) + "\n"
