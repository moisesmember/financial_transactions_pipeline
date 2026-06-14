"""Immutable training-run history and comparison index."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any
from uuid import uuid4

import pandas as pd

from src.config.settings import Settings
from src.models.governance_artifacts import write_manifest
from src.utils.logger import get_logger


logger = get_logger(__name__)


class TrainingHistoryRegistry:
    """Persist each completed training run without replacing prior runs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def new_run_id(model_name: str) -> str:
        """Create a sortable and collision-resistant training run ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        normalized_model = model_name.strip().lower().replace(" ", "_")
        return f"{timestamp}_{normalized_model}_{uuid4().hex[:8]}"

    def record(
        self,
        run_id: str,
        metadata: dict[str, Any],
        leakage_report: dict[str, Any],
        started_at: datetime,
        completed_at: datetime,
    ) -> Path:
        """Save one immutable run directory and update the comparison index."""
        run_dir = self.settings.training_history_dir / run_id
        if run_dir.exists():
            raise FileExistsError(f"Historico de treino ja existe: {run_dir}")
        run_dir.mkdir(parents=True)

        files = {
            self.settings.artifact_path(filename): run_dir / filename
            for filename in self.settings.governance_artifact_filenames
            if self.settings.artifact_path(filename).exists()
            and (
                filename != self.settings.pipeline_filename
                or self.settings.training_history_save_pipeline
            )
        }
        for source, target in files.items():
            copy2(source, target)

        pipeline_hash = self._sha256(self.settings.pipeline_path)
        historical_metadata = {
            **metadata,
            "run": {
                "run_id": run_id,
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": completed_at.isoformat(),
                "duration_seconds": (completed_at - started_at).total_seconds(),
                "pipeline_sha256": pipeline_hash,
                "pipeline_archived": self.settings.training_history_save_pipeline,
                "status": metadata.get("status", "completed"),
            },
        }
        metadata_json_path = run_dir / "metadata.json"
        metadata_json_path.write_text(
            json.dumps(self._json_safe(historical_metadata), indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        write_manifest(
            run_dir / self.settings.manifest_filename,
            list(run_dir.iterdir()),
        )
        self._append_index(run_dir, historical_metadata, leakage_report)
        logger.info(
            "Historico de treino gravado | run_id=%s | diretorio=%s",
            run_id,
            run_dir,
        )
        return run_dir

    def _append_index(
        self,
        run_dir: Path,
        metadata: dict[str, Any],
        leakage_report: dict[str, Any],
    ) -> None:
        run = metadata["run"]
        validation = metadata["validation_metrics"]
        test = metadata["test_metrics"]
        out_of_time = metadata.get("out_of_time_metrics", {})
        selection = metadata["threshold_selection"]
        row = {
            "run_id": run["run_id"],
            "completed_at_utc": run["completed_at_utc"],
            "duration_seconds": run["duration_seconds"],
            "model_name": metadata["model_name"],
            "threshold": metadata["threshold"],
            "threshold_strategy": selection["strategy"],
            "false_positive_cost": selection["false_positive_cost"],
            "false_negative_cost": selection["false_negative_cost"],
            "validation_precision": validation["precision"],
            "validation_recall": validation["recall"],
            "validation_f1": validation["f1"],
            "validation_fbeta": validation["fbeta"],
            "validation_pr_auc": validation["pr_auc"],
            "validation_roc_auc": validation["roc_auc"],
            "validation_tp": validation["tp"],
            "validation_fp": validation["fp"],
            "validation_fn": validation["fn"],
            "validation_business_cost": metadata["operational_costs"]["validation"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_fbeta": test["fbeta"],
            "test_pr_auc": test["pr_auc"],
            "test_roc_auc": test["roc_auc"],
            "test_tp": test["tp"],
            "test_fp": test["fp"],
            "test_fn": test["fn"],
            "test_business_cost": metadata["operational_costs"]["test"],
            "out_of_time_precision": out_of_time.get("precision"),
            "out_of_time_recall": out_of_time.get("recall"),
            "out_of_time_fbeta": out_of_time.get("fbeta"),
            "out_of_time_pr_auc": out_of_time.get("pr_auc"),
            "out_of_time_roc_auc": out_of_time.get("roc_auc"),
            "out_of_time_alert_rate": out_of_time.get("alert_rate"),
            "out_of_time_business_cost": metadata["operational_costs"].get("out_of_time"),
            "audit_status": leakage_report["status"],
            "audit_warning_count": len(leakage_report.get("warnings", [])),
            "audit_failure_count": len(leakage_report.get("failures", [])),
            "training_max_rows": metadata["training_max_rows"],
            "train_rows": metadata["dataset"]["train_rows"],
            "validation_rows": metadata["dataset"]["validation_rows"],
            "test_rows": metadata["dataset"]["test_rows"],
            "out_of_time_rows": metadata["dataset"].get("out_of_time_rows"),
            "train_positive_rate": metadata["dataset"]["train_positive_rate"],
            "validation_positive_rate": metadata["dataset"]["validation_positive_rate"],
            "test_positive_rate": metadata["dataset"]["test_positive_rate"],
            "out_of_time_positive_rate": metadata["dataset"].get("out_of_time_positive_rate"),
            "strict_leakage_prevention": metadata["strict_leakage_prevention"],
            "pipeline_sha256": run["pipeline_sha256"],
            "dataset_version": metadata.get("dataset_version"),
            "feature_set_version": metadata.get("feature_set_version"),
            "code_version": metadata.get("code_version"),
            "experiment_fingerprint": metadata.get("experiment_fingerprint"),
            "promotion_decision": metadata.get("baseline_decision", {}).get("decision"),
            "run_directory": run_dir.relative_to(self.settings.artifacts_dir).as_posix(),
        }

        index_path = self.settings.training_history_index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        new_row = pd.DataFrame([row])
        if index_path.exists():
            index = pd.read_csv(index_path)
            index = pd.concat([index, new_row], ignore_index=True)
        else:
            index = new_row
        index = index.drop_duplicates(subset=["run_id"], keep="last").sort_values(
            "completed_at_utc",
            ascending=False,
        )
        temporary_path = index_path.with_suffix(".tmp")
        index.to_csv(temporary_path, index=False)
        temporary_path.replace(index_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
