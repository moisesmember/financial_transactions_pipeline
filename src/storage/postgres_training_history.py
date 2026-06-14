"""Optional PostgreSQL persistence for completed training runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, create_engine, inspect, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)
SCHEMA = "fraud_tracking"
REQUIRED_TABLES = {
    "training_runs",
    "run_metrics",
    "threshold_evaluations",
    "run_artifacts",
}
OPTIONAL_TABLES = {
    "baseline_promotions",
    "leakage_audit_checks",
    "model_features",
    "robustness_experiments",
    "model_search_trials",
    "external_benchmark_results",
}


class PostgresTrainingHistoryRepository:
    """Persist historical run files into the migrated PostgreSQL schema."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def persist_if_available(self, run_dir: Path) -> bool:
        """Persist a run when PostgreSQL and all required tables are available."""
        if not self.settings.database_tracking_enabled:
            logger.info("Persistencia PostgreSQL desabilitada por configuracao.")
            return False

        run_dir = run_dir.resolve()
        engine = create_engine(
            self.settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": self.settings.database_connect_timeout_seconds},
        )
        try:
            with engine.connect() as connection:
                inspector = inspect(connection)
                available_tables = set(inspector.get_table_names(schema=SCHEMA))
                missing_tables = REQUIRED_TABLES - available_tables
                if missing_tables:
                    logger.info(
                        "Historico PostgreSQL ignorado | schema/tabelas ausentes=%s",
                        ",".join(sorted(missing_tables)),
                    )
                    return False

            metadata = MetaData(schema=SCHEMA)
            tables_to_reflect = set(REQUIRED_TABLES) | (OPTIONAL_TABLES & available_tables)
            metadata.reflect(bind=engine, only=sorted(tables_to_reflect))
            historical_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            leakage_report = json.loads(
                (run_dir / self.settings.leakage_report_filename).read_text(encoding="utf-8")
            )
            thresholds = pd.read_csv(run_dir / self.settings.threshold_analysis_filename)
            scenario_path = run_dir / self.settings.threshold_cost_scenarios_filename
            if scenario_path.exists():
                thresholds = pd.concat(
                    [thresholds, pd.read_csv(scenario_path)],
                    ignore_index=True,
                ).drop_duplicates(subset=["scenario_name", "split", "threshold"], keep="last")

            with engine.begin() as connection:
                self._insert_run(connection, metadata, run_dir, historical_metadata, leakage_report)
                self._insert_metrics(connection, metadata, historical_metadata)
                self._insert_thresholds(connection, metadata, historical_metadata["run"]["run_id"], thresholds)
                self._insert_artifacts(connection, metadata, historical_metadata["run"]["run_id"], run_dir)
                self._insert_leakage_checks(
                    connection,
                    metadata,
                    historical_metadata["run"]["run_id"],
                    leakage_report,
                )
                self._insert_model_features(
                    connection,
                    metadata,
                    historical_metadata["run"]["run_id"],
                    run_dir,
                )
                self._insert_robustness_experiments(connection, metadata, run_dir)
                self._insert_model_search_trials(
                    connection,
                    metadata,
                    historical_metadata["run"]["run_id"],
                    run_dir,
                )
                self._insert_external_benchmarks(
                    connection,
                    metadata,
                    historical_metadata["run"]["run_id"],
                    run_dir,
                )
                decision = historical_metadata.get("baseline_decision", {}).get("decision")
                if (
                    self.settings.promote_baseline
                    and decision == "promote"
                    and "baseline_promotions" in available_tables
                ):
                    self._insert_baseline_promotion(
                        connection,
                        metadata,
                        historical_metadata,
                        leakage_report,
                    )

            logger.info(
                "Historico persistido no PostgreSQL | run_id=%s",
                historical_metadata["run"]["run_id"],
            )
            return True
        except (OSError, SQLAlchemyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Historico PostgreSQL indisponivel; treino preservado localmente | erro=%s",
                exc,
            )
            return False
        finally:
            engine.dispose()

    def _insert_run(
        self,
        connection,
        metadata: MetaData,
        run_dir: Path,
        historical_metadata: dict[str, Any],
        leakage_report: dict[str, Any],
    ) -> None:
        run = historical_metadata["run"]
        dataset = historical_metadata["dataset"]
        selection = historical_metadata["threshold_selection"]
        table = metadata.tables[f"{SCHEMA}.training_runs"]
        values = self._supported_values(
            table,
            {
                "run_id": run["run_id"],
                "started_at": run["started_at_utc"],
                "completed_at": run["completed_at_utc"],
                "duration_seconds": run["duration_seconds"],
                "model_name": historical_metadata["model_name"],
                "selected_threshold": historical_metadata["threshold"],
                "threshold_strategy": selection["strategy"],
                "false_positive_cost": selection["false_positive_cost"],
                "false_negative_cost": selection["false_negative_cost"],
                "audit_status": leakage_report["status"],
                "audit_warning_count": len(leakage_report.get("warnings", [])),
                "audit_failure_count": len(leakage_report.get("failures", [])),
                "training_max_rows": historical_metadata["training_max_rows"],
                "train_rows": dataset["train_rows"],
                "validation_rows": dataset["validation_rows"],
                "test_rows": dataset["test_rows"],
                "out_of_time_rows": dataset.get("out_of_time_rows"),
                "train_positive_rate": dataset["train_positive_rate"],
                "validation_positive_rate": dataset["validation_positive_rate"],
                "test_positive_rate": dataset["test_positive_rate"],
                "out_of_time_positive_rate": dataset.get("out_of_time_positive_rate"),
                "strict_leakage_prevention": historical_metadata["strict_leakage_prevention"],
                "pipeline_sha256": run["pipeline_sha256"],
                "run_directory": run_dir.relative_to(self.settings.artifacts_dir).as_posix(),
                "metadata": historical_metadata,
                "leakage_audit": leakage_report,
                "status": historical_metadata.get("status", "completed"),
                "dataset_version": historical_metadata.get("dataset_version"),
                "dataset_sha256": historical_metadata.get("dataset_sha256"),
                "feature_set_version": historical_metadata.get("feature_set_version"),
                "code_version": historical_metadata.get("code_version"),
                "experiment_fingerprint": historical_metadata.get("experiment_fingerprint"),
                "promotion_decision": historical_metadata.get("baseline_decision", {}).get(
                    "decision"
                ),
                "promotion_reason": historical_metadata.get("baseline_decision", {}).get(
                    "reasons"
                ),
                "model_selection_engine": historical_metadata.get("model_selection", {}).get(
                    "engine"
                ),
                "model_selection_objective": historical_metadata.get(
                    "model_selection", {}
                ).get("objective"),
                "model_selection_trial_count": historical_metadata.get(
                    "model_selection", {}
                ).get("trial_count"),
            },
        )
        statement = insert(table).values(**values)
        updates = {
            key: getattr(statement.excluded, key)
            for key in values
            if key not in {"run_id", "created_at"}
        }
        connection.execute(
            statement.on_conflict_do_update(index_elements=["run_id"], set_=updates)
        )

    @staticmethod
    def _insert_metrics(connection, metadata: MetaData, historical_metadata: dict[str, Any]) -> None:
        table = metadata.tables[f"{SCHEMA}.run_metrics"]
        run_id = historical_metadata["run"]["run_id"]
        costs = historical_metadata["operational_costs"]
        rows = []
        for split in ("validation", "test", "out_of_time"):
            metrics = historical_metadata.get(f"{split}_metrics")
            if metrics is None:
                continue
            rows.append(
                PostgresTrainingHistoryRepository._supported_values(table, {
                    "run_id": run_id,
                    "split": split,
                    "threshold": metrics["threshold"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fbeta": metrics["fbeta"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "tn": int(metrics["tn"]),
                    "fn": int(metrics["fn"]),
                    "business_cost": costs[split],
                    "alerts": int(metrics.get("alerts", metrics["tp"] + metrics["fp"])),
                    "alert_rate": metrics.get("alert_rate"),
                    "cost_per_record": costs[split]
                    / max(1, metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]),
                })
            )
        statement = insert(table).values(rows)
        updates = {
            column.name: getattr(statement.excluded, column.name)
            for column in table.columns
            if column.name not in {"run_id", "split", "created_at"}
        }
        connection.execute(
            statement.on_conflict_do_update(index_elements=["run_id", "split"], set_=updates)
        )

    @staticmethod
    def _insert_thresholds(
        connection,
        metadata: MetaData,
        run_id: str,
        thresholds: pd.DataFrame,
    ) -> None:
        table = metadata.tables[f"{SCHEMA}.threshold_evaluations"]
        rows = []
        for record in thresholds.to_dict(orient="records"):
            rows.append(
                PostgresTrainingHistoryRepository._supported_values(table, {
                    "run_id": run_id,
                    "scenario_name": str(record.get("scenario_name", "primary")),
                    "split": str(record["split"]),
                    "threshold": float(record["threshold"]),
                    "precision": float(record["precision"]),
                    "recall": float(record["recall"]),
                    "f1": float(record["f1"]),
                    "fbeta": float(record["fbeta"]),
                    "tp": int(record["tp"]),
                    "fp": int(record["fp"]),
                    "tn": int(record["tn"]),
                    "fn": int(record["fn"]),
                    "alerts": int(record["alerts"]),
                    "alert_rate": float(record["alert_rate"]),
                    "business_cost": float(record["business_cost"]),
                    "cost_per_record": float(record["cost_per_record"]),
                    "false_positive_cost": float(record.get("false_positive_cost", 0)),
                    "false_negative_cost": float(record.get("false_negative_cost", 0)),
                })
            )
        statement = insert(table).values(rows)
        conflict_columns = ["run_id", "split", "threshold"]
        if "scenario_name" in table.c:
            conflict_columns.insert(1, "scenario_name")
        updates = {
            column.name: getattr(statement.excluded, column.name)
            for column in table.columns
            if column.name not in {*conflict_columns, "created_at"}
        }
        connection.execute(
            statement.on_conflict_do_update(index_elements=conflict_columns, set_=updates)
        )

    def _insert_artifacts(
        self,
        connection,
        metadata: MetaData,
        run_id: str,
        run_dir: Path,
    ) -> None:
        table = metadata.tables[f"{SCHEMA}.run_artifacts"]
        rows = []
        for path in sorted(item for item in run_dir.iterdir() if item.is_file()):
            rows.append(
                {
                    "run_id": run_id,
                    "artifact_type": self._artifact_type(path),
                    "uri": path.relative_to(self.settings.project_root).as_posix(),
                    "sha256": self._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_nothing(
                index_elements=["run_id", "artifact_type", "uri"]
            )
        )

    @staticmethod
    def _insert_leakage_checks(
        connection,
        metadata: MetaData,
        run_id: str,
        leakage_report: dict[str, Any],
    ) -> None:
        key = f"{SCHEMA}.leakage_audit_checks"
        if key not in metadata.tables:
            return
        table = metadata.tables[key]
        rows = [
            {
                "run_id": run_id,
                "check_name": check["check_name"],
                "check_result": check["check_result"],
                "severity": check["severity"],
                "message": check.get("message"),
                "recommendation": check.get("recommendation"),
            }
            for check in leakage_report.get("check_results", [])
        ]
        if not rows:
            return
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["run_id", "check_name"],
                set_={
                    "check_result": statement.excluded.check_result,
                    "severity": statement.excluded.severity,
                    "message": statement.excluded.message,
                    "recommendation": statement.excluded.recommendation,
                },
            )
        )

    def _insert_model_features(
        self,
        connection,
        metadata: MetaData,
        run_id: str,
        run_dir: Path,
    ) -> None:
        key = f"{SCHEMA}.model_features"
        path = run_dir / self.settings.feature_importance_filename
        if key not in metadata.tables or not path.exists():
            return
        table = metadata.tables[key]
        frame = pd.read_csv(path).where(pd.notna, None)
        rows = [
            self._supported_values(table, {"run_id": run_id, **record})
            for record in frame.to_dict(orient="records")
        ]
        if not rows:
            return
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["run_id", "feature_name"],
                set_={
                    column.name: getattr(statement.excluded, column.name)
                    for column in table.columns
                    if column.name not in {"id", "run_id", "feature_name", "created_at"}
                },
            )
        )

    def _insert_robustness_experiments(
        self,
        connection,
        metadata: MetaData,
        run_dir: Path,
    ) -> None:
        key = f"{SCHEMA}.robustness_experiments"
        path = run_dir / self.settings.geo_ablation_filename
        if key not in metadata.tables or not path.exists():
            return
        table = metadata.tables[key]
        frame = pd.read_csv(path)
        rows = []
        for record in frame.to_dict(orient="records"):
            record["features_removed"] = json.loads(record["features_removed"])
            record["top_features"] = json.loads(record["top_features"])
            rows.append(self._supported_values(table, record))
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["experiment_run_id"],
                set_={
                    column.name: getattr(statement.excluded, column.name)
                    for column in table.columns
                    if column.name != "experiment_run_id"
                },
            )
        )

    def _insert_model_search_trials(
        self,
        connection,
        metadata: MetaData,
        run_id: str,
        run_dir: Path,
    ) -> None:
        key = f"{SCHEMA}.model_search_trials"
        path = run_dir / self.settings.optuna_trials_filename
        if key not in metadata.tables or not path.exists() or path.stat().st_size == 0:
            return
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return
        if frame.empty:
            return
        table = metadata.tables[key]
        rows = []
        for record in frame.where(pd.notna(frame), None).to_dict(orient="records"):
            raw_params = record.get("model_params")
            record["model_params"] = json.loads(raw_params) if raw_params else {}
            rows.append(self._supported_values(table, {"run_id": run_id, **record}))
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["run_id", "trial_number"],
                set_={
                    column.name: getattr(statement.excluded, column.name)
                    for column in table.columns
                    if column.name not in {"run_id", "trial_number", "created_at"}
                },
            )
        )

    def _insert_external_benchmarks(
        self,
        connection,
        metadata: MetaData,
        run_id: str,
        run_dir: Path,
    ) -> None:
        key = f"{SCHEMA}.external_benchmark_results"
        if key not in metadata.tables:
            return
        table = metadata.tables[key]
        rows: list[dict[str, Any]] = []
        results_path = run_dir / self.settings.external_benchmark_filename
        if results_path.exists() and results_path.stat().st_size > 0:
            try:
                results = pd.read_csv(results_path)
            except pd.errors.EmptyDataError:
                results = pd.DataFrame()
            for record in results.where(pd.notna(results), None).to_dict(orient="records"):
                rows.append(
                    self._supported_values(
                        table,
                        {"run_id": run_id, "status": "completed", **record},
                    )
                )
        summary_path = run_dir / self.settings.external_benchmark_summary_filename
        if summary_path.exists():
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
            for summary in summaries:
                backend = summary.get("backend")
                if not backend:
                    continue
                rows.append(
                    self._supported_values(
                        table,
                        {
                            "run_id": run_id,
                            "backend": backend,
                            "split": "summary",
                            "status": summary.get("status", "unknown"),
                            "framework_model": summary.get("framework_model"),
                            "duration_seconds": summary.get("duration_seconds"),
                            "message": summary.get("message"),
                            "metadata": summary.get("metadata"),
                        },
                    )
                )
        if not rows:
            return
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_update(
                index_elements=["run_id", "backend", "split"],
                set_={
                    column.name: getattr(statement.excluded, column.name)
                    for column in table.columns
                    if column.name not in {"run_id", "backend", "split", "created_at"}
                },
            )
        )

    @staticmethod
    def _insert_baseline_promotion(
        connection,
        metadata: MetaData,
        historical_metadata: dict[str, Any],
        leakage_report: dict[str, Any],
    ) -> None:
        table = metadata.tables[f"{SCHEMA}.baseline_promotions"]
        previous_baseline_run_id = connection.execute(
            select(table.c.run_id).where(table.c.is_active.is_(True))
        ).scalar_one_or_none()
        connection.execute(update(table).where(table.c.is_active.is_(True)).values(is_active=False))
        existing_id = connection.execute(
            select(table.c.id).where(
                table.c.run_id == historical_metadata["run"]["run_id"],
                table.c.pipeline_sha256 == historical_metadata["run"]["pipeline_sha256"],
            )
        ).scalar_one_or_none()
        if existing_id is not None:
            connection.execute(
                update(table).where(table.c.id == existing_id).values(is_active=True)
            )
            return
        values = PostgresTrainingHistoryRepository._supported_values(
            table,
            {
                "run_id": historical_metadata["run"]["run_id"],
                "promoted_at": datetime.now(timezone.utc),
                "audit_status": leakage_report["status"],
                "pipeline_sha256": historical_metadata["run"]["pipeline_sha256"],
                "is_active": True,
                "metadata": historical_metadata,
                "previous_baseline_run_id": previous_baseline_run_id,
                "decision": "promote",
                "decision_reason": historical_metadata.get("baseline_decision", {}).get("reasons"),
                "approval_status": "approved",
                "rollback_available": True,
            },
        )
        statement = insert(table).values(**values)
        connection.execute(statement)
        runs = metadata.tables[f"{SCHEMA}.training_runs"]
        connection.execute(
            update(runs)
            .where(runs.c.run_id == historical_metadata["run"]["run_id"])
            .values(status="promoted")
        )

    def _artifact_type(self, path: Path) -> str:
        mapping = {
            self.settings.pipeline_filename: "pipeline",
            self.settings.metadata_filename: "metadata_joblib",
            "metadata.json": "metadata_json",
            self.settings.threshold_analysis_filename: "threshold_analysis",
            self.settings.leakage_report_filename: "leakage_audit",
            self.settings.threshold_cost_scenarios_filename: "threshold_cost_scenarios",
            self.settings.feature_importance_filename: "feature_importance",
            self.settings.calibration_report_filename: "calibration_report",
            self.settings.score_deciles_filename: "score_deciles",
            self.settings.calibration_metrics_filename: "calibration_metrics",
            self.settings.calibration_curve_filename: "calibration_curve",
            self.settings.out_of_time_metrics_filename: "out_of_time_metrics",
            self.settings.model_card_filename: "model_card",
            self.settings.baseline_decision_filename: "baseline_decision",
            self.settings.manifest_filename: "manifest",
            self.settings.geo_ablation_filename: "geo_ablation",
            self.settings.optuna_trials_filename: "optuna_trials",
            self.settings.optuna_study_filename: "optuna_study",
            self.settings.external_benchmark_filename: "external_benchmark_results",
            self.settings.external_benchmark_summary_filename: "external_benchmark_summary",
        }
        return mapping.get(path.name, "other")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _supported_values(table, values: dict[str, Any]) -> dict[str, Any]:
        """Filter payloads to columns available in the migrated database."""
        return {key: value for key, value in values.items() if key in table.c}
