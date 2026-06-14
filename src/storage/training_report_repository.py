"""Read all persisted governance data required by a training report."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import Settings
from src.storage.model_run_fact_repository import (
    QUALIFIED_VIEW,
    SCHEMA,
    VIEW,
    ModelRunFactViewNotFoundError,
    _json_safe,
)


class TrainingReportRepositoryError(RuntimeError):
    """Raised when the training report cannot be read."""


class TrainingReportNotFoundError(TrainingReportRepositoryError):
    """Raised when the requested model run does not exist."""


class TrainingReportRepository:
    """Load one run and its normalized feature, audit and search records."""

    def __init__(self, settings: Settings, engine: Engine | None = None) -> None:
        self._engine = engine or create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": settings.database_connect_timeout_seconds},
        )

    def get(self, run_id: str, feature_limit: int) -> dict[str, Any]:
        """Return raw persisted report sections for one run."""
        try:
            with self._engine.connect() as connection:
                inspector = inspect(connection)
                if VIEW not in set(inspector.get_view_names(schema=SCHEMA)):
                    raise ModelRunFactViewNotFoundError(
                        f"View {QUALIFIED_VIEW} nao encontrada."
                    )
                run = connection.execute(
                    text(f"SELECT * FROM {QUALIFIED_VIEW} WHERE run_id = :run_id"),
                    {"run_id": run_id},
                ).mappings().one_or_none()
                if run is None:
                    raise TrainingReportNotFoundError(run_id)

                available_tables = set(inspector.get_table_names(schema=SCHEMA))
                return {
                    "run": _json_safe(dict(run)),
                    "feature_count": self._feature_count(
                        connection, available_tables, run_id
                    ),
                    "features": self._rows(
                        connection,
                        available_tables,
                        "model_features",
                        """
                        SELECT feature_name, importance, absolute_importance, direction,
                               odds_ratio, feature_group, is_geo_feature,
                               is_temporal_feature, is_behavioral_feature, is_risk_feature
                        FROM fraud_tracking.model_features
                        WHERE run_id = :run_id
                        ORDER BY absolute_importance DESC, feature_name
                        LIMIT :feature_limit
                        """,
                        {"run_id": run_id, "feature_limit": feature_limit},
                    ),
                    "audit_checks": self._rows(
                        connection,
                        available_tables,
                        "leakage_audit_checks",
                        """
                        SELECT check_name, check_result, severity, message, recommendation
                        FROM fraud_tracking.leakage_audit_checks
                        WHERE run_id = :run_id
                        ORDER BY CASE severity WHEN 'critical' THEN 0 ELSE 1 END, check_name
                        """,
                        {"run_id": run_id},
                    ),
                    "search_trials": self._rows(
                        connection,
                        available_tables,
                        "model_search_trials",
                        """
                        SELECT trial_number, state, model_name, model_params,
                               validation_pr_auc, selected_threshold, precision, recall,
                               alert_rate, business_cost, duration_seconds
                        FROM fraud_tracking.model_search_trials
                        WHERE run_id = :run_id
                        ORDER BY validation_pr_auc DESC NULLS LAST, trial_number
                        """,
                        {"run_id": run_id},
                    ),
                    "benchmarks": self._rows(
                        connection,
                        available_tables,
                        "external_benchmark_results",
                        """
                        SELECT backend, split, status, framework_model, threshold,
                               precision, recall, f1, fbeta, pr_auc, roc_auc,
                               tp, fp, tn, fn, alerts, alert_rate, business_cost,
                               duration_seconds, message, metadata
                        FROM fraud_tracking.external_benchmark_results
                        WHERE run_id = :run_id
                        ORDER BY backend, split
                        """,
                        {"run_id": run_id},
                    ),
                    "robustness": self._rows(
                        connection,
                        available_tables,
                        "robustness_experiments",
                        """
                        SELECT experiment_run_id, experiment_group, feature_set_version,
                               features_removed, threshold, precision, recall, f1, fbeta,
                               pr_auc, roc_auc, tp, fp, tn, fn, business_cost,
                               alert_rate, top_features
                        FROM fraud_tracking.robustness_experiments
                        WHERE parent_run_id = :run_id
                        ORDER BY experiment_group
                        """,
                        {"run_id": run_id},
                    ),
                }
        except (ModelRunFactViewNotFoundError, TrainingReportNotFoundError):
            raise
        except SQLAlchemyError as exc:
            raise TrainingReportRepositoryError(
                f"Nao foi possivel montar o report do treino {run_id}."
            ) from exc

    @staticmethod
    def _rows(
        connection,
        available_tables: set[str],
        table_name: str,
        query: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if table_name not in available_tables:
            return []
        rows = connection.execute(text(query), parameters).mappings()
        return [_json_safe(dict(row)) for row in rows]

    @staticmethod
    def _feature_count(
        connection,
        available_tables: set[str],
        run_id: str,
    ) -> int:
        if "model_features" not in available_tables:
            return 0
        return int(
            connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM fraud_tracking.model_features
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).scalar_one()
        )
