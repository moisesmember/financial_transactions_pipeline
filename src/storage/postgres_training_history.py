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
            tables_to_reflect = set(REQUIRED_TABLES)
            if "baseline_promotions" in available_tables:
                tables_to_reflect.add("baseline_promotions")
            metadata.reflect(bind=engine, only=sorted(tables_to_reflect))
            historical_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            leakage_report = json.loads(
                (run_dir / self.settings.leakage_report_filename).read_text(encoding="utf-8")
            )
            thresholds = pd.read_csv(run_dir / self.settings.threshold_analysis_filename)

            with engine.begin() as connection:
                self._insert_run(connection, metadata, run_dir, historical_metadata, leakage_report)
                self._insert_metrics(connection, metadata, historical_metadata)
                self._insert_thresholds(connection, metadata, historical_metadata["run"]["run_id"], thresholds)
                self._insert_artifacts(connection, metadata, historical_metadata["run"]["run_id"], run_dir)
                if self.settings.promote_baseline and "baseline_promotions" in available_tables:
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
        statement = insert(table).values(
            run_id=run["run_id"],
            started_at=run["started_at_utc"],
            completed_at=run["completed_at_utc"],
            duration_seconds=run["duration_seconds"],
            model_name=historical_metadata["model_name"],
            selected_threshold=historical_metadata["threshold"],
            threshold_strategy=selection["strategy"],
            false_positive_cost=selection["false_positive_cost"],
            false_negative_cost=selection["false_negative_cost"],
            audit_status=leakage_report["status"],
            audit_warning_count=len(leakage_report.get("warnings", [])),
            training_max_rows=historical_metadata["training_max_rows"],
            train_rows=dataset["train_rows"],
            validation_rows=dataset["validation_rows"],
            test_rows=dataset["test_rows"],
            train_positive_rate=dataset["train_positive_rate"],
            validation_positive_rate=dataset["validation_positive_rate"],
            test_positive_rate=dataset["test_positive_rate"],
            strict_leakage_prevention=historical_metadata["strict_leakage_prevention"],
            pipeline_sha256=run["pipeline_sha256"],
            run_directory=run_dir.relative_to(self.settings.artifacts_dir).as_posix(),
            metadata=historical_metadata,
            leakage_audit=leakage_report,
        )
        connection.execute(statement.on_conflict_do_nothing(index_elements=["run_id"]))

    @staticmethod
    def _insert_metrics(connection, metadata: MetaData, historical_metadata: dict[str, Any]) -> None:
        table = metadata.tables[f"{SCHEMA}.run_metrics"]
        run_id = historical_metadata["run"]["run_id"]
        costs = historical_metadata["operational_costs"]
        rows = []
        for split in ("validation", "test"):
            metrics = historical_metadata[f"{split}_metrics"]
            rows.append(
                {
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
                }
            )
        statement = insert(table).values(rows)
        connection.execute(statement.on_conflict_do_nothing(index_elements=["run_id", "split"]))

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
                {
                    "run_id": run_id,
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
                }
            )
        statement = insert(table).values(rows)
        connection.execute(
            statement.on_conflict_do_nothing(index_elements=["run_id", "split", "threshold"])
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
    def _insert_baseline_promotion(
        connection,
        metadata: MetaData,
        historical_metadata: dict[str, Any],
        leakage_report: dict[str, Any],
    ) -> None:
        table = metadata.tables[f"{SCHEMA}.baseline_promotions"]
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
        statement = insert(table).values(
            run_id=historical_metadata["run"]["run_id"],
            promoted_at=datetime.now(timezone.utc),
            audit_status=leakage_report["status"],
            pipeline_sha256=historical_metadata["run"]["pipeline_sha256"],
            is_active=True,
            metadata=historical_metadata,
        )
        connection.execute(statement)

    def _artifact_type(self, path: Path) -> str:
        mapping = {
            self.settings.pipeline_filename: "pipeline",
            self.settings.metadata_filename: "metadata_joblib",
            "metadata.json": "metadata_json",
            self.settings.threshold_analysis_filename: "threshold_analysis",
            self.settings.leakage_report_filename: "leakage_audit",
        }
        return mapping.get(path.name, "other")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
