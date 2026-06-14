"""End-to-end training pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config.settings import Settings
from src.data.limit_data import TrainingDataLimiter
from src.data.load_data import RawDataRepository
from src.data.merge_data import FraudDataMerger
from src.data.split_data import TemporalSplitter
from src.features.cleaning import FraudDataCleaner
from src.ingestion.import_service import DatasetImportService
from src.models.baseline import BaselineRegistry
from src.models.baseline_decision import BaselineDecisionService
from src.models.calibration import write_calibration_artifacts
from src.models.evaluate import evaluate_binary_classifier
from src.models.feature_report import build_feature_importance
from src.models.governance_artifacts import write_manifest, write_model_card
from src.models.leakage_audit import LeakageAuditService
from src.models.robustness import run_geographic_ablation
from src.models.threshold import find_best_threshold
from src.models.threshold_analysis import (
    build_cost_scenario_summary,
    build_threshold_table,
    select_business_threshold,
    threshold_grid,
)
from src.models.train import FraudModelTrainer
from src.models.training_history import TrainingHistoryRegistry
from src.models.versioning import code_version, dataset_fingerprint, experiment_fingerprint
from src.storage.postgres_training_history import PostgresTrainingHistoryRepository
from src.storage.sync import StorageSyncService
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Training output metadata."""

    model_name: str
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    out_of_time_metrics: dict[str, float]
    pipeline_path: Path
    metadata_path: Path
    threshold_analysis_path: Path
    leakage_report_path: Path
    run_id: str
    history_run_dir: Path
    baseline_decision: str


class TrainingPipeline:
    """Service layer that executes the full training workflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> TrainingResult:
        """Load data, merge, split, train, tune threshold, evaluate and persist."""
        started_at = datetime.now(timezone.utc)
        run_id = TrainingHistoryRegistry.new_run_id(self.settings.model_name)
        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.kaggle_auto_import:
            DatasetImportService(self.settings).import_data()

        repository = RawDataRepository(self.settings)
        raw = repository.load_all()
        raw["transactions"] = TrainingDataLimiter(self.settings.training_max_rows).apply(raw["transactions"])
        merged = FraudDataMerger(self.settings).merge(
            transactions=raw["transactions"],
            cards=raw["cards"],
            users=raw["users"],
            mcc_codes=raw["mcc"],
            labels=raw["labels"],
        )
        del raw

        cleaned_for_split = FraudDataCleaner(self.settings).fit_transform(merged)
        splits = TemporalSplitter(self.settings).split(cleaned_for_split)

        X_train, y_train = self._split_xy(splits.train)
        X_val, y_val = self._split_xy(splits.validation)
        X_test, y_test = self._split_xy(splits.test)
        if splits.out_of_time is None:
            raise ValueError("O split out-of-time e obrigatorio para o treinamento governado.")
        X_out_of_time, y_out_of_time = self._split_xy(splits.out_of_time)

        pipeline = FraudModelTrainer(self.settings).train(X_train, y_train)
        validation_scores = self._predict_scores(pipeline, X_val)
        thresholds = threshold_grid(
            self.settings.threshold_analysis_start,
            self.settings.threshold_analysis_stop,
            self.settings.threshold_analysis_step,
        )
        validation_table = build_threshold_table(
            y_val.to_numpy(),
            validation_scores,
            thresholds=thresholds,
            beta=self.settings.threshold_beta,
            false_positive_cost=self.settings.false_positive_cost,
            false_negative_cost=self.settings.false_negative_cost,
            split="validation",
        )
        threshold, threshold_metrics = self._select_threshold(
            validation_table,
            y_val,
            validation_scores,
        )
        logger.info(
            "Threshold escolhido na validacao | estrategia=%s | threshold=%.4f | %s",
            self.settings.threshold_selection_strategy,
            threshold,
            threshold_metrics,
        )

        test_scores = self._predict_scores(pipeline, X_test)
        out_of_time_scores = self._predict_scores(pipeline, X_out_of_time)
        test_table = build_threshold_table(
            y_test.to_numpy(),
            test_scores,
            thresholds=thresholds,
            beta=self.settings.threshold_beta,
            false_positive_cost=self.settings.false_positive_cost,
            false_negative_cost=self.settings.false_negative_cost,
            split="test",
        )
        out_of_time_table = build_threshold_table(
            y_out_of_time.to_numpy(),
            out_of_time_scores,
            thresholds=thresholds,
            beta=self.settings.threshold_beta,
            false_positive_cost=self.settings.false_positive_cost,
            false_negative_cost=self.settings.false_negative_cost,
            split="out_of_time",
        )
        pd.concat(
            [validation_table, test_table, out_of_time_table],
            ignore_index=True,
        ).to_csv(
            self.settings.threshold_analysis_path,
            index=False,
        )
        cost_scenarios = build_cost_scenario_summary(
            {
                "validation": (y_val.to_numpy(), validation_scores),
                "test": (y_test.to_numpy(), test_scores),
                "out_of_time": (y_out_of_time.to_numpy(), out_of_time_scores),
            },
            thresholds=thresholds,
            beta=self.settings.threshold_beta,
            cost_scenarios=self.settings.threshold_cost_scenarios,
        )
        cost_scenarios.to_csv(
            self.settings.artifact_path(self.settings.threshold_cost_scenarios_filename),
            index=False,
        )
        validation_metrics = evaluate_binary_classifier(
            y_val.to_numpy(), validation_scores, threshold=threshold, beta=self.settings.threshold_beta
        )
        test_metrics = evaluate_binary_classifier(
            y_test.to_numpy(), test_scores, threshold=threshold, beta=self.settings.threshold_beta
        )
        out_of_time_metrics = evaluate_binary_classifier(
            y_out_of_time.to_numpy(),
            out_of_time_scores,
            threshold=threshold,
            beta=self.settings.threshold_beta,
        )
        leakage_report = LeakageAuditService(self.settings).build_report(
            splits,
            pipeline,
            validation_metrics,
            test_metrics,
            selected_threshold=threshold,
            out_of_time_metrics=out_of_time_metrics,
        )
        self.settings.leakage_report_path.write_text(
            json.dumps(leakage_report, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )

        joblib.dump(pipeline, self.settings.pipeline_path)
        feature_importance = build_feature_importance(pipeline)
        feature_importance.to_csv(
            self.settings.artifact_path(self.settings.feature_importance_filename),
            index=False,
        )
        if self.settings.run_geo_ablation:
            primary_ablation_metrics = {
                **test_metrics,
                "business_cost": (
                    test_metrics["fp"] * self.settings.false_positive_cost
                    + test_metrics["fn"] * self.settings.false_negative_cost
                ),
            }
            run_geographic_ablation(
                self.settings,
                run_id,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                primary_ablation_metrics,
                feature_importance,
            ).to_csv(
                self.settings.artifact_path(self.settings.geo_ablation_filename),
                index=False,
            )
        write_calibration_artifacts(
            {
                "validation": (y_val.to_numpy(), validation_scores),
                "test": (y_test.to_numpy(), test_scores),
                "out_of_time": (y_out_of_time.to_numpy(), out_of_time_scores),
            },
            report_path=self.settings.artifact_path(self.settings.calibration_report_filename),
            deciles_path=self.settings.artifact_path(self.settings.score_deciles_filename),
            metrics_path=self.settings.artifact_path(self.settings.calibration_metrics_filename),
            curve_path=self.settings.artifact_path(self.settings.calibration_curve_filename),
        )
        self.settings.artifact_path(self.settings.out_of_time_metrics_filename).write_text(
            json.dumps(out_of_time_metrics, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        dataset_version = dataset_fingerprint(self.settings)
        current_code_version = code_version(self.settings)
        reproducibility = {
            "dataset_version": dataset_version,
            "feature_set_version": self.settings.feature_set_version,
            "code_version": current_code_version,
            "model_name": self.settings.model_name,
            "model_params": self.settings.model_params[self.settings.model_name],
            "random_state": self.settings.random_state,
            "training_max_rows": self.settings.training_max_rows,
        }
        metadata = {
            "run_id": run_id,
            "status": "completed",
            "model_name": self.settings.model_name,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "out_of_time_metrics": out_of_time_metrics,
            "time_column": splits.time_column,
            "training_max_rows": self.settings.training_max_rows,
            "strict_leakage_prevention": self.settings.strict_leakage_prevention,
            "dataset": {
                "train_rows": len(y_train),
                "validation_rows": len(y_val),
                "test_rows": len(y_test),
                "out_of_time_rows": len(y_out_of_time),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
                "test_positive_rate": float(y_test.mean()),
                "out_of_time_positive_rate": float(y_out_of_time.mean()),
                "train_time_min": splits.train[splits.time_column].min().isoformat(),
                "train_time_max": splits.train[splits.time_column].max().isoformat(),
                "validation_time_min": splits.validation[splits.time_column].min().isoformat(),
                "validation_time_max": splits.validation[splits.time_column].max().isoformat(),
                "test_time_min": splits.test[splits.time_column].min().isoformat(),
                "test_time_max": splits.test[splits.time_column].max().isoformat(),
                "out_of_time_time_min": splits.out_of_time[splits.time_column].min().isoformat(),
                "out_of_time_time_max": splits.out_of_time[splits.time_column].max().isoformat(),
            },
            "threshold_selection": {
                "strategy": self.settings.threshold_selection_strategy,
                "false_positive_cost": self.settings.false_positive_cost,
                "false_negative_cost": self.settings.false_negative_cost,
                "analysis_start": self.settings.threshold_analysis_start,
                "analysis_stop": self.settings.threshold_analysis_stop,
                "analysis_step": self.settings.threshold_analysis_step,
            },
            "operational_costs": {
                "validation": (
                    validation_metrics["fp"] * self.settings.false_positive_cost
                    + validation_metrics["fn"] * self.settings.false_negative_cost
                ),
                "test": (
                    test_metrics["fp"] * self.settings.false_positive_cost
                    + test_metrics["fn"] * self.settings.false_negative_cost
                ),
                "out_of_time": (
                    out_of_time_metrics["fp"] * self.settings.false_positive_cost
                    + out_of_time_metrics["fn"] * self.settings.false_negative_cost
                ),
            },
            "leakage_audit_status": leakage_report["status"],
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_version,
            "feature_set_version": self.settings.feature_set_version,
            "code_version": current_code_version,
            "experiment_fingerprint": experiment_fingerprint(reproducibility),
            "geo_ablation_enabled": self.settings.run_geo_ablation,
        }
        required_for_decision = [
            self.settings.pipeline_path,
            self.settings.threshold_analysis_path,
            self.settings.leakage_report_path,
            self.settings.artifact_path(self.settings.threshold_cost_scenarios_filename),
            self.settings.artifact_path(self.settings.feature_importance_filename),
            self.settings.artifact_path(self.settings.calibration_report_filename),
            self.settings.artifact_path(self.settings.score_deciles_filename),
            self.settings.artifact_path(self.settings.out_of_time_metrics_filename),
        ]
        baseline_decision = BaselineDecisionService(self.settings).decide(
            metadata,
            leakage_report,
            required_for_decision,
        )
        metadata["baseline_decision"] = baseline_decision
        if baseline_decision["decision"] == "reject":
            metadata["status"] = "rejected"
        joblib.dump(metadata, self.settings.metadata_path)
        self.settings.artifact_path(self.settings.baseline_decision_filename).write_text(
            json.dumps(baseline_decision, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        write_model_card(
            self.settings.artifact_path(self.settings.model_card_filename),
            metadata,
            leakage_report,
            baseline_decision,
        )
        manifest_path = self.settings.artifact_path(self.settings.manifest_filename)
        write_manifest(
            manifest_path,
            [
                self.settings.artifact_path(filename)
                for filename in self.settings.governance_artifact_filenames
            ],
        )
        completed_at = datetime.now(timezone.utc)
        history_run_dir = TrainingHistoryRegistry(self.settings).record(
            run_id=run_id,
            metadata=metadata,
            leakage_report=leakage_report,
            started_at=started_at,
            completed_at=completed_at,
        )
        logger.info("Pipeline e metadados salvos em %s", self.settings.artifacts_dir)
        baseline_registry: BaselineRegistry | None = None
        if self.settings.promote_baseline and baseline_decision["decision"] == "promote":
            baseline_registry = BaselineRegistry(self.settings)
            baseline_registry.promote(
                metadata,
                report_paths=[
                    self.settings.artifact_path(filename)
                    for filename in self.settings.governance_artifact_filenames
                    if filename not in {self.settings.pipeline_filename, self.settings.metadata_filename}
                ],
                overwrite=self.settings.baseline_overwrite,
                audit_status=leakage_report["status"],
            )
        elif self.settings.promote_baseline:
            logger.warning(
                "Promocao de baseline bloqueada | decisao=%s | motivos=%s",
                baseline_decision["decision"],
                "; ".join(baseline_decision["reasons"]),
            )
        persisted = PostgresTrainingHistoryRepository(self.settings).persist_if_available(
            history_run_dir
        )
        if baseline_registry is not None:
            if self.settings.database_tracking_enabled and not persisted:
                baseline_registry.rollback_promotion()
                logger.error(
                    "Promocao local revertida porque o registro PostgreSQL nao foi concluido."
                )
            else:
                baseline_registry.commit_promotion()
        StorageSyncService(self.settings).upload_artifacts(history_run_dir=history_run_dir)

        return TrainingResult(
            model_name=self.settings.model_name,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            out_of_time_metrics=out_of_time_metrics,
            pipeline_path=self.settings.pipeline_path,
            metadata_path=self.settings.metadata_path,
            threshold_analysis_path=self.settings.threshold_analysis_path,
            leakage_report_path=self.settings.leakage_report_path,
            run_id=run_id,
            history_run_dir=history_run_dir,
            baseline_decision=baseline_decision["decision"],
        )

    def _select_threshold(
        self,
        validation_table: pd.DataFrame,
        y_val: pd.Series,
        validation_scores,
    ) -> tuple[float, dict[str, float]]:
        """Select the operational threshold using the configured strategy."""
        if self.settings.threshold_selection_strategy == "business_cost":
            return select_business_threshold(validation_table)
        return find_best_threshold(
            y_val.to_numpy(),
            validation_scores,
            beta=self.settings.threshold_beta,
            min_precision=self.settings.min_precision_for_threshold,
        )

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target."""
        if self.settings.target_column not in df.columns:
            raise ValueError(f"Coluna alvo ausente: {self.settings.target_column}")
        y = df[self.settings.target_column].astype(int)
        X = df.drop(columns=[self.settings.target_column])
        return X, y

    @staticmethod
    def _predict_scores(pipeline, X: pd.DataFrame):
        """Return positive-class probabilities or decision scores."""
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X)[:, 1]
        scores = pipeline.decision_function(X)
        return 1 / (1 + pd.Series(-scores).map(lambda value: pow(2.718281828, value))).to_numpy()
