"""Automated checks for common temporal and feature leakage risks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.data.split_data import DataSplits
from src.utils.logger import get_logger


logger = get_logger(__name__)

SNAPSHOT_RISK_COLUMNS = {
    "card_on_dark_web",
    "credit_limit",
    "credit_score",
    "current_age",
    "num_cards_issued",
    "num_credit_cards",
    "per_capita_income",
    "retirement_age",
    "total_debt",
    "year_pin_last_changed",
    "yearly_income",
}
POST_EVENT_RISK_COLUMNS = {"errors"}
SENSITIVE_COLUMNS = {"address", "card_number", "cvv"}


class LeakageAuditService:
    """Audit split boundaries and model inputs for leakage indicators."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_report(
        self,
        splits: DataSplits,
        pipeline: Any,
        validation_metrics: dict[str, float],
        test_metrics: dict[str, float],
        selected_threshold: float,
        out_of_time_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return a serializable audit report with status and recommendations."""
        time_col = splits.time_column
        temporal_order_valid = bool(
            splits.train[time_col].max() < splits.validation[time_col].min()
            and splits.validation[time_col].max() < splits.test[time_col].min()
            and (
                splits.out_of_time is None
                or splits.test[time_col].max() < splits.out_of_time[time_col].min()
            )
        )
        duplicate_ids = self._duplicate_ids(splits)
        selected_columns = self._selected_input_columns(pipeline)
        excluded_columns = self._excluded_input_columns(pipeline)
        snapshot_risks = sorted(selected_columns & SNAPSHOT_RISK_COLUMNS)
        post_event_risks = sorted(selected_columns & POST_EVENT_RISK_COLUMNS)
        sensitive_risks = sorted(selected_columns & SENSITIVE_COLUMNS)
        direct_target_selected = self.settings.target_column in selected_columns
        high_auc = max(
            validation_metrics.get("roc_auc", 0.0),
            test_metrics.get("roc_auc", 0.0),
            (out_of_time_metrics or {}).get("roc_auc", 0.0),
        ) >= self.settings.leakage_roc_auc_warning
        threshold_at_boundary = bool(
            np.isclose(selected_threshold, self.settings.threshold_analysis_start)
            or np.isclose(selected_threshold, self.settings.threshold_analysis_stop)
        )
        limited_training_window = self.settings.training_max_rows is not None
        top_model_features = self._top_model_features(pipeline)
        top_five_names = [str(item["feature"]).lower() for item in top_model_features[:5]]
        geographic_dominance = sum(
            any(token in name for token in ("merchant_city", "merchant_state", "zip", "latitude", "longitude"))
            for name in top_five_names
        ) >= 2

        failures: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []
        if not temporal_order_valid:
            failures.append("Os splits temporais se sobrepoem ou estao fora de ordem.")
        if duplicate_ids:
            failures.append("Existem IDs de transacao repetidos entre os splits.")
        if direct_target_selected:
            failures.append("A coluna alvo foi selecionada como feature.")
        if snapshot_risks:
            warnings.append(
                "Features de snapshot podem conter informacao posterior a transacao: "
                + ", ".join(snapshot_risks)
            )
            recommendations.append(
                "Executar um treino de ablation sem features de snapshot e comparar PR-AUC."
            )
        if post_event_risks:
            warnings.append(
                "Features potencialmente conhecidas somente apos a decisao: "
                + ", ".join(post_event_risks)
            )
            recommendations.append(
                "Confirmar se essas features existem no instante real da autorizacao."
            )
        if sensitive_risks:
            warnings.append("Features sensiveis selecionadas: " + ", ".join(sensitive_risks))
            recommendations.append("Remover PII e credenciais do conjunto de features.")
        if high_auc:
            warnings.append(
                f"ROC-AUC maior ou igual a {self.settings.leakage_roc_auc_warning:.3f}."
            )
            recommendations.append(
                "Validar em uma janela temporal futura e revisar as features mais importantes."
            )
        if threshold_at_boundary:
            warnings.append("O threshold selecionado esta no limite da faixa analisada.")
            recommendations.append(
                "Ampliar THRESHOLD_ANALYSIS_START/STOP antes de aprovar o custo operacional."
            )
        if limited_training_window:
            warnings.append(
                "O treino usa uma amostra limitada de transacoes ao longo do horizonte."
            )
            recommendations.append(
                "Confirmar os resultados com maior volume ou com o dataset completo."
            )
        if geographic_dominance:
            warnings.append("Features geograficas dominam pelo menos duas das cinco maiores importancias.")
            recommendations.append(
                "Executar os experimentos de ablation geografica antes de promover o modelo."
            )

        status = "fail" if failures else "warning" if warnings else "pass"
        check_results = [
            self._check("temporal_order", temporal_order_valid, "critical"),
            self._check("duplicate_transaction_ids", duplicate_ids == 0, "critical"),
            self._check("target_not_selected", not direct_target_selected, "critical"),
            self._check("roc_auc_below_warning", not high_auc, "warning"),
            self._check("threshold_inside_analysis_range", not threshold_at_boundary, "warning"),
            self._check("full_training_horizon", not limited_training_window, "warning"),
            self._check("geographic_features_not_dominant", not geographic_dominance, "warning"),
        ]
        report = {
            "status": status,
            "checks": {
                "temporal_order_valid": temporal_order_valid,
                "duplicate_transaction_ids_across_splits": duplicate_ids,
                "target_selected_as_feature": direct_target_selected,
                "high_roc_auc_warning": high_auc,
                "threshold_at_analysis_boundary": threshold_at_boundary,
                "limited_training_window": limited_training_window,
                "geographic_feature_dominance": geographic_dominance,
            },
            "split_boundaries": {
                "train_max": splits.train[time_col].max().isoformat(),
                "validation_min": splits.validation[time_col].min().isoformat(),
                "validation_max": splits.validation[time_col].max().isoformat(),
                "test_min": splits.test[time_col].min().isoformat(),
                "test_max": splits.test[time_col].max().isoformat(),
                "out_of_time_min": (
                    splits.out_of_time[time_col].min().isoformat()
                    if splits.out_of_time is not None
                    else None
                ),
            },
            "selected_input_columns": sorted(selected_columns),
            "excluded_input_columns": sorted(excluded_columns),
            "top_model_features": top_model_features,
            "risk_columns": {
                "snapshot": snapshot_risks,
                "post_event": post_event_risks,
                "sensitive": sensitive_risks,
            },
            "failures": failures,
            "warnings": warnings,
            "recommendations": recommendations,
            "check_results": check_results,
        }
        log_method = logger.error if failures else logger.warning if warnings else logger.info
        log_method(
            "Auditoria de leakage concluida | status=%s | falhas=%d | alertas=%d",
            status,
            len(failures),
            len(warnings),
        )
        return report

    def _duplicate_ids(self, splits: DataSplits) -> int:
        id_column = next(
            (
                candidate
                for candidate in self.settings.transaction_id_candidates
                if candidate in splits.train.columns
            ),
            None,
        )
        if id_column is None:
            return 0
        train_ids = set(splits.train[id_column].astype(str))
        validation_ids = set(splits.validation[id_column].astype(str))
        test_ids = set(splits.test[id_column].astype(str))
        partitions = [train_ids, validation_ids, test_ids]
        if splits.out_of_time is not None:
            partitions.append(set(splits.out_of_time[id_column].astype(str)))
        return sum(
            len(partitions[left] & partitions[right])
            for left in range(len(partitions))
            for right in range(left + 1, len(partitions))
        )

    @staticmethod
    def _check(name: str, passed: bool, severity: str) -> dict[str, Any]:
        return {
            "check_name": name,
            "check_result": "pass" if passed else "fail",
            "severity": severity,
        }

    @staticmethod
    def _selected_input_columns(pipeline: Any) -> set[str]:
        preprocessor = pipeline.named_steps["preprocessor"]
        selected: set[str] = set()
        for name, _, columns in preprocessor.transformers_:
            if name == "drop":
                continue
            selected.update(str(column) for column in columns)
        return selected

    @staticmethod
    def _excluded_input_columns(pipeline: Any) -> set[str]:
        """Return columns explicitly dropped by the fitted preprocessor."""
        preprocessor = pipeline.named_steps["preprocessor"]
        excluded: set[str] = set()
        for name, _, columns in preprocessor.transformers_:
            if name == "drop":
                excluded.update(str(column) for column in columns)
        return excluded

    @staticmethod
    def _top_model_features(pipeline: Any, limit: int = 20) -> list[dict[str, float | str]]:
        """Return the strongest linear coefficients or tree importances."""
        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")
        if model is None or preprocessor is None:
            return []
        try:
            names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            return []

        if hasattr(model, "coef_"):
            values = np.asarray(model.coef_)
            values = values[0] if values.ndim > 1 else values
        elif hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        else:
            return []
        if len(values) != len(names):
            return []

        indexes = np.argsort(np.abs(values))[-limit:][::-1]
        return [
            {
                "feature": str(names[index]),
                "importance": float(values[index]),
                "absolute_importance": float(abs(values[index])),
            }
            for index in indexes
        ]
