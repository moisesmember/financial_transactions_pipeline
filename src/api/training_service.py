"""In-process orchestration for asynchronous model training jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from src.api.schemas import TrainingRequest
from src.config.settings import Settings
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)


class TrainingAlreadyRunningError(RuntimeError):
    """Raised when another training job already owns the process resources."""


class TrainingJobNotFoundError(KeyError):
    """Raised when a requested training job does not exist."""


class TrainingJobManager:
    """Run one resource-intensive training job at a time in a worker thread."""

    def __init__(self, on_complete: Callable[[], None] | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fraud-training")
        self._lock = Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job_id: str | None = None
        self._on_complete = on_complete

    def start(self, request: TrainingRequest) -> dict[str, Any]:
        """Queue a training job using .env values for omitted request properties."""
        job_id = uuid4().hex
        settings = replace(Settings(), **request.settings_overrides())
        settings = replace(
            settings,
            artifacts_dir=(
                settings.project_root / ".runtime" / "training" / job_id / "artifacts"
            ),
        )
        job = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "started_at": None,
            "completed_at": None,
            "configuration": self._training_configuration(settings),
            "result": None,
            "error": None,
        }
        with self._lock:
            if self._active_job_id is not None:
                raise TrainingAlreadyRunningError(self._active_job_id)
            self._active_job_id = job_id
            self._jobs[job_id] = job
            self._executor.submit(self._run, job_id, settings)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        """Return a snapshot of one training job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise TrainingJobNotFoundError(job_id)
            return {
                **job,
                "configuration": dict(job["configuration"]),
                "result": dict(job["result"]) if job["result"] is not None else None,
            }

    def close(self) -> None:
        """Release the worker thread after tests or controlled application shutdown."""
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _run(self, job_id: str, settings: Settings) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = datetime.now(timezone.utc)
        try:
            result = TrainingPipeline(settings).run()
            result_payload = {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "threshold": result.threshold,
                "baseline_decision": result.baseline_decision,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
                "out_of_time_metrics": result.out_of_time_metrics,
                "artifact_location": settings.object_uri(settings.pipeline_object_key),
                "history_location": settings.object_uri(
                    settings.artifact_object_key(f"history/{result.run_id}")
                ),
            }
            with self._lock:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = result_payload
            if self._on_complete is not None:
                self._on_complete()
        except Exception as exc:  # noqa: BLE001 - job state must capture all failures
            logger.exception("Treinamento iniciado pela API falhou | job_id=%s", job_id)
            with self._lock:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = str(exc)
        finally:
            with self._lock:
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc)
                self._active_job_id = None

    @staticmethod
    def _training_configuration(settings: Settings) -> dict[str, Any]:
        """Expose resolved non-secret controls used by the training run."""
        names = (
            "threshold_selection_strategy",
            "threshold_analysis_start",
            "threshold_analysis_stop",
            "threshold_analysis_step",
            "false_positive_cost",
            "false_negative_cost",
            "threshold_cost_scenarios",
            "out_of_time_size",
            "leakage_roc_auc_warning",
            "strict_leakage_prevention",
            "promote_baseline",
            "baseline_overwrite",
            "run_geo_ablation",
            "training_history_save_pipeline",
            "training_max_rows",
            "baseline_warning_justification",
            "promotion_min_recall",
            "promotion_max_alert_rate",
            "promotion_max_oot_pr_auc_drop",
            "promotion_max_cost_increase",
        )
        values = asdict(settings)
        return {name: values[name] for name in names}
