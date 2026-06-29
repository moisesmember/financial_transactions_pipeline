"""Optional MLflow experiment tracking for governed training runs."""

from __future__ import annotations

import math
from inspect import signature as inspect_signature
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class MlflowTrackingService:
    """Log completed training runs to MLflow without making training depend on MLflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def log_completed_run(
        self,
        run_dir: Path,
        metadata: dict[str, Any],
        leakage_report: dict[str, Any],
        input_example: pd.DataFrame | None = None,
        prediction_example: Any | None = None,
    ) -> str | None:
        """Log params, metrics, tags, artifacts and optionally the sklearn model."""
        if not self.settings.mlflow_tracking_enabled:
            logger.info("Tracking MLflow desabilitado por configuracao.")
            return None

        try:
            import mlflow
        except ImportError:
            logger.warning(
                "Tracking MLflow habilitado, mas o pacote mlflow nao esta instalado."
            )
            return None

        run_dir = run_dir.resolve()
        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            experiment_id = self._resolve_experiment_id(mlflow)
            run_id = metadata["run_id"]
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_id,
                tags=self._tags(metadata, leakage_report),
            ) as active_run:
                mlflow.log_params(self._params(metadata))
                mlflow.log_metrics(self._metrics(metadata))
                self._log_governance_artifacts(mlflow, run_dir)
                try:
                    self._log_model(
                        mlflow,
                        run_dir,
                        input_example=input_example,
                        prediction_example=prediction_example,
                    )
                except Exception as exc:  # noqa: BLE001 - model artifact must not hide run metrics
                    logger.warning(
                        "Modelo MLflow nao registrado; params, metricas e artefatos "
                        "governados foram preservados quando disponiveis | erro=%s",
                        exc,
                    )
                mlflow_run_id = active_run.info.run_id
            logger.info(
                "Run registrado no MLflow | training_run_id=%s | mlflow_run_id=%s",
                run_id,
                mlflow_run_id,
            )
            return mlflow_run_id
        except Exception as exc:  # noqa: BLE001 - MLflow must not fail governed training
            logger.warning(
                "Tracking MLflow indisponivel; treino preservado no historico local | erro=%s",
                exc,
            )
            return None

    @staticmethod
    def _log_governance_artifacts(mlflow, run_dir: Path) -> None:
        """Log governed artifacts without making MLflow model logging all-or-nothing."""
        try:
            mlflow.log_artifacts(str(run_dir), artifact_path="governance")
        except Exception as exc:  # noqa: BLE001 - metrics/params are still useful
            logger.warning(
                "Artefatos governados nao foram registrados no MLflow | diretorio=%s | erro=%s",
                run_dir,
                exc,
            )

    def _resolve_experiment_id(self, mlflow) -> str:
        experiment = mlflow.get_experiment_by_name(self.settings.mlflow_experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return mlflow.create_experiment(
            name=self.settings.mlflow_experiment_name,
            artifact_location=self.settings.mlflow_artifact_location,
        )

    def _log_model(
        self,
        mlflow,
        run_dir: Path,
        input_example: pd.DataFrame | None,
        prediction_example: Any | None,
    ) -> None:
        if not self.settings.mlflow_log_model:
            return
        pipeline_path = run_dir / self.settings.pipeline_filename
        if not pipeline_path.exists():
            logger.info("Modelo MLflow nao registrado; pipeline historica ausente.")
            return

        model = joblib.load(pipeline_path)
        model_input_example = self._prepare_input_example(input_example)
        signature = None
        if model_input_example is not None and prediction_example is not None:
            try:
                mlflow_models = getattr(mlflow, "models", None) or import_module("mlflow.models")
                signature = mlflow_models.infer_signature(
                    model_input_example,
                    prediction_example,
                )
            except Exception as exc:  # noqa: BLE001 - signature is useful, not mandatory
                logger.info("Assinatura MLflow ignorada | erro=%s", exc)

        registered_model_name = (
            self.settings.mlflow_registered_model_name
            if self.settings.mlflow_register_model
            else None
        )
        mlflow_sklearn = getattr(mlflow, "sklearn", None) or import_module("mlflow.sklearn")
        log_model_signature = inspect_signature(mlflow_sklearn.log_model)
        model_location_argument = (
            {"name": "model"}
            if "name" in log_model_signature.parameters
            else {"artifact_path": "model"}
        )
        mlflow_sklearn.log_model(
            sk_model=model,
            **model_location_argument,
            signature=signature,
            input_example=model_input_example,
            registered_model_name=registered_model_name,
            serialization_format="cloudpickle",
        )

    def _params(self, metadata: dict[str, Any]) -> dict[str, str]:
        threshold_selection = metadata.get("threshold_selection", {})
        model_selection = metadata.get("model_selection", {})
        params: dict[str, Any] = {
            "model_name": metadata.get("model_name"),
            "model_selection_engine": model_selection.get("engine"),
            "model_selection_objective": model_selection.get("objective"),
            "model_selection_trial_count": model_selection.get("trial_count"),
            "optuna_temporal_holdout_fraction": model_selection.get("temporal_holdout_fraction"),
            "optuna_pr_auc_stability_penalty": model_selection.get("pr_auc_stability_penalty"),
            "threshold_strategy": threshold_selection.get("strategy"),
            "false_positive_cost": threshold_selection.get("false_positive_cost"),
            "false_negative_cost": threshold_selection.get("false_negative_cost"),
            "threshold_analysis_start": threshold_selection.get("analysis_start"),
            "threshold_analysis_stop": threshold_selection.get("analysis_stop"),
            "threshold_analysis_step": threshold_selection.get("analysis_step"),
            "feature_set_version": metadata.get("feature_set_version"),
            "code_version": metadata.get("code_version"),
            "dataset_version": metadata.get("dataset_version"),
            "training_max_rows": metadata.get("training_max_rows"),
            "strict_leakage_prevention": metadata.get("strict_leakage_prevention"),
            "geo_ablation_enabled": metadata.get("geo_ablation_enabled"),
            "feature_exclusions": ",".join(metadata.get("feature_exclusions", [])),
            "exclude_geographic_features": metadata.get("exclude_geographic_features"),
            "leakage_audit_status": metadata.get("leakage_audit_status"),
            "target_audit_status": metadata.get("target_audit_status"),
            "data_drift_status": metadata.get("data_drift_status"),
            "robustness_status": metadata.get("robustness_status"),
            "walk_forward_status": metadata.get("walk_forward_status"),
            "baseline_decision": metadata.get("baseline_decision", {}).get("decision"),
        }
        for key, value in metadata.get("model_params", {}).items():
            params[f"model_param_{key}"] = value
        return {
            key: self._param_value(value)
            for key, value in params.items()
            if value is not None
        }

    def _metrics(self, metadata: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {
            "selected_threshold": metadata["threshold"],
        }
        for split in ("validation", "test", "out_of_time"):
            for key, value in metadata.get(f"{split}_metrics", {}).items():
                metrics[f"{split}_{key}"] = value
            cost = metadata.get("operational_costs", {}).get(split)
            if cost is not None:
                metrics[f"{split}_business_cost"] = cost

        dataset = metadata.get("dataset", {})
        for key in (
            "train_rows",
            "validation_rows",
            "test_rows",
            "out_of_time_rows",
            "train_positive_rate",
            "validation_positive_rate",
            "test_positive_rate",
            "out_of_time_positive_rate",
        ):
            if key in dataset:
                metrics[f"dataset_{key}"] = dataset[key]

        return {
            key: float(value)
            for key, value in metrics.items()
            if self._is_number(value)
        }

    @staticmethod
    def _tags(metadata: dict[str, Any], leakage_report: dict[str, Any]) -> dict[str, str]:
        return {
            "training_run_id": str(metadata["run_id"]),
            "status": str(metadata.get("status", "completed")),
            "audit_status": str(leakage_report.get("status", "unknown")),
            "target_audit_status": str(metadata.get("target_audit_status", "unknown")),
            "data_drift_status": str(metadata.get("data_drift_status", "unknown")),
            "robustness_status": str(metadata.get("robustness_status", "unknown")),
            "walk_forward_status": str(metadata.get("walk_forward_status", "unknown")),
            "promotion_decision": str(metadata.get("baseline_decision", {}).get("decision")),
            "experiment_fingerprint": str(metadata.get("experiment_fingerprint")),
        }

    @staticmethod
    def _param_value(value: Any) -> str:
        text = str(value)
        if len(text) > 500:
            return text[:497] + "..."
        return text

    @staticmethod
    def _prepare_input_example(input_example: pd.DataFrame | None) -> pd.DataFrame | None:
        if input_example is None:
            return None
        prepared = input_example.copy()
        integer_columns = prepared.select_dtypes(include=["integer"]).columns
        if len(integer_columns) > 0:
            prepared[integer_columns] = prepared[integer_columns].astype("float64")
        return prepared

    @staticmethod
    def _is_number(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value))
