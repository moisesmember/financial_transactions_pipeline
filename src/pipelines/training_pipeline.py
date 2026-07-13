"""End-to-end training pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.models.data_drift import DataDriftReportService
from src.models.evaluate import evaluate_binary_classifier
from src.models.error_attribution import build_error_attribution_report
from src.models.external_benchmarks import BENCHMARK_RESULT_COLUMNS, ExternalBenchmarkRunner
from src.models.feature_report import build_feature_importance
from src.models.governance_artifacts import (
    write_manifest,
    write_model_card,
    write_model_review_report,
)
from src.models.leakage_audit import LeakageAuditService
from src.models.mlflow_tracking import MlflowTrackingService
from src.models.optuna_search import OptunaModelSelector
from src.models.robustness import run_geographic_ablation, write_robustness_reports
from src.models.target_audit import TargetAuditService
from src.models.threshold import find_best_threshold
from src.models.threshold_analysis import (
    build_threshold_recommendations,
    build_cost_scenario_summary,
    build_threshold_table,
    select_business_threshold,
    threshold_grid,
)
from src.models.train import FraudModelTrainer
from src.models.training_history import TrainingHistoryRegistry
from src.models.versioning import code_version, dataset_fingerprint, experiment_fingerprint
from src.models.walk_forward import WalkForwardValidator
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
    mlflow_run_id: str | None
    history_run_dir: Path
    baseline_decision: str


class TrainingPipeline:
    """Service layer that executes the full training workflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> TrainingResult:
        """Load data, merge, split, train, tune threshold, evaluate and persist."""
        started_at = datetime.now(timezone.utc)
        storage_sync = StorageSyncService(self.settings)
        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        storage_sync.prepare_artifact_workspace()
        if self.settings.kaggle_auto_import:
            DatasetImportService(self.settings).import_data()

        repository = RawDataRepository(self.settings)
        raw = repository.load_all()
        raw_transactions = raw["transactions"].copy()
        raw_labels = raw["labels"]
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
        target_audit = TargetAuditService(self.settings).build(
            raw_transactions,
            raw_labels,
            splits,
            self.settings.artifacts_dir,
        )

        X_train, y_train = self._split_xy(splits.train)
        X_val, y_val = self._split_xy(splits.validation)
        X_test, y_test = self._split_xy(splits.test)
        if splits.out_of_time is None:
            raise ValueError("O split out-of-time e obrigatorio para o treinamento governado.")
        X_out_of_time, y_out_of_time = self._split_xy(splits.out_of_time)

        if self.settings.model_selection_engine == "optuna":
            selection = OptunaModelSelector(self.settings).select(
                X_train,
                y_train,
                X_val,
                y_val,
                trials_path=self.settings.artifact_path(self.settings.optuna_trials_filename),
                study_path=self.settings.artifact_path(self.settings.optuna_study_filename),
            )
            pipeline = selection.pipeline
            selected_model_name = selection.model_name
            selected_model_params = selection.model_params
            selection_metadata = {
                "engine": "optuna",
                "objective": self.settings.optuna_selection_objective,
                "validation_pr_auc": selection.validation_pr_auc,
                "trial_count": selection.trial_count,
                "temporal_holdout_fraction": self.settings.optuna_temporal_holdout_fraction,
                "pr_auc_stability_penalty": self.settings.optuna_pr_auc_stability_penalty,
                "recall_stability_penalty": self.settings.optuna_recall_stability_penalty,
                "last_window_penalty": self.settings.optuna_last_window_penalty,
            }
        else:
            selected_model_name = self.settings.model_name
            selected_model_params = self.settings.model_params[selected_model_name]
            pipeline = FraudModelTrainer(self.settings).train(
                X_train,
                y_train,
                model_name=selected_model_name,
                model_params=selected_model_params,
            )
            pd.DataFrame().to_csv(
                self.settings.artifact_path(self.settings.optuna_trials_filename),
                index=False,
            )
            self.settings.artifact_path(self.settings.optuna_study_filename).write_text(
                json.dumps(
                    {
                        "engine": "fixed",
                        "model_name": selected_model_name,
                        "model_params": selected_model_params,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            selection_metadata = {"engine": "fixed", "trial_count": 0}
        run_id = TrainingHistoryRegistry.new_run_id(selected_model_name)
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

        train_scores = self._predict_scores(pipeline, X_train)
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
        threshold_analysis = pd.concat(
            [validation_table, test_table, out_of_time_table],
            ignore_index=True,
        )
        threshold_analysis.to_csv(
            self.settings.threshold_analysis_path,
            index=False,
        )
        threshold_recommendations = build_threshold_recommendations(
            threshold_analysis,
            max_alert_rate=self.settings.promotion_max_alert_rate,
        )
        self.settings.artifact_path(self.settings.threshold_recommendations_filename).write_text(
            json.dumps(
                threshold_recommendations,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ),
            encoding="utf-8",
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
        train_metrics = evaluate_binary_classifier(
            y_train.to_numpy(), train_scores, threshold=threshold, beta=self.settings.threshold_beta
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

        original_train_rows = len(splits.train)
        original_train_positives = int(splits.train[self.settings.target_column].eq(1).sum())
        limited_train = TrainingDataLimiter(
            self.settings.training_max_rows,
            negative_positive_ratio=self.settings.training_negative_positive_ratio,
            random_state=self.settings.random_state,
        ).apply(
            splits.train,
            target_column=self.settings.target_column,
            time_column=splits.time_column,
        )
        splits = type(splits)(
            train=limited_train,
            validation=splits.validation,
            test=splits.test,
            time_column=splits.time_column,
            out_of_time=splits.out_of_time,
        )
        build_error_attribution_report(
            {"validation": X_val, "test": X_test, "out_of_time": X_out_of_time},
            {"validation": y_val, "test": y_test, "out_of_time": y_out_of_time},
            {
                "validation": validation_scores,
                "test": test_scores,
                "out_of_time": out_of_time_scores,
            },
            threshold,
            self.settings.artifact_path(self.settings.error_attribution_report_filename),
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
        important_features = feature_importance["feature_name"].head(30).tolist() if not feature_importance.empty else []
        drift_report = DataDriftReportService(self.settings).build(
            splits,
            self.settings.artifacts_dir,
            important_features=important_features,
        )
        metrics_by_split = {
            "train": {
                **train_metrics,
                "business_cost": (
                    train_metrics["fp"] * self.settings.false_positive_cost
                    + train_metrics["fn"] * self.settings.false_negative_cost
                ),
            },
            "validation": {
                **validation_metrics,
                "business_cost": (
                    validation_metrics["fp"] * self.settings.false_positive_cost
                    + validation_metrics["fn"] * self.settings.false_negative_cost
                ),
            },
            "test": {
                **test_metrics,
                "business_cost": (
                    test_metrics["fp"] * self.settings.false_positive_cost
                    + test_metrics["fn"] * self.settings.false_negative_cost
                ),
            },
            "out_of_time": {
                **out_of_time_metrics,
                "business_cost": (
                    out_of_time_metrics["fp"] * self.settings.false_positive_cost
                    + out_of_time_metrics["fn"] * self.settings.false_negative_cost
                ),
            },
        }
        if self.settings.run_geo_ablation:
            robustness_results = run_geographic_ablation(
                self.settings,
                run_id,
                selected_model_name,
                selected_model_params,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                X_out_of_time,
                y_out_of_time,
                metrics_by_split,
                feature_importance,
            )
        else:
            robustness_results = pd.DataFrame()
        robustness_results.to_csv(
            self.settings.artifact_path(self.settings.geo_ablation_filename),
            index=False,
        )
        robustness_report = write_robustness_reports(
            robustness_results,
            self.settings.artifacts_dir,
            self.settings,
        )
        walk_forward_report = WalkForwardValidator(self.settings).run(
            cleaned_for_split,
            splits.time_column,
            selected_model_name,
            selected_model_params,
            self.settings.artifacts_dir,
        )
        benchmark_results_path = self.settings.artifact_path(
            self.settings.external_benchmark_filename
        )
        benchmark_summary_path = self.settings.artifact_path(
            self.settings.external_benchmark_summary_filename
        )
        external_benchmark_summary = self._run_external_benchmarks(
            pipeline,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_out_of_time,
            y_out_of_time,
            run_id,
            benchmark_results_path,
            benchmark_summary_path,
        )
        try:
            external_rows = pd.read_csv(benchmark_results_path)
        except pd.errors.EmptyDataError:
            external_rows = pd.DataFrame()
        internal_rows = pd.DataFrame(
            [
                {
                    "backend": "optuna_sklearn",
                    "framework_model": selected_model_name,
                    "split": split,
                    **metrics,
                    "business_cost": (
                        metrics["fp"] * self.settings.false_positive_cost
                        + metrics["fn"] * self.settings.false_negative_cost
                    ),
                }
                for split, metrics in (
                    ("validation", validation_metrics),
                    ("test", test_metrics),
                    ("out_of_time", out_of_time_metrics),
                )
            ]
        )
        benchmark_rows = internal_rows
        if not external_rows.empty:
            benchmark_rows = pd.concat(
                [internal_rows, external_rows.dropna(axis=1, how="all")],
                ignore_index=True,
            )
        benchmark_rows.to_csv(
            benchmark_results_path,
            index=False,
        )
        external_benchmark_summary.insert(
            0,
            {
                "backend": "optuna_sklearn",
                "status": "completed",
                "framework_model": selected_model_name,
                "selection_engine": self.settings.model_selection_engine,
            },
        )
        benchmark_summary_path.write_text(
            json.dumps(
                external_benchmark_summary,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
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
            "model_name": selected_model_name,
            "model_params": selected_model_params,
            "random_state": self.settings.random_state,
            "training_max_rows": self.settings.training_max_rows,
            "training_negative_positive_ratio": self.settings.training_negative_positive_ratio,
            "feature_exclusions": self.settings.feature_exclusions,
            "exclude_geographic_features": self.settings.exclude_geographic_features,
            "optuna_selection_objective": self.settings.optuna_selection_objective,
        }
        metadata = {
            "run_id": run_id,
            "status": "completed",
            "model_name": selected_model_name,
            "model_params": selected_model_params,
            "model_selection": selection_metadata,
            "external_benchmarks": external_benchmark_summary,
            "threshold": threshold,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "out_of_time_metrics": out_of_time_metrics,
            "time_column": splits.time_column,
            "training_max_rows": self.settings.training_max_rows,
            "training_negative_positive_ratio": self.settings.training_negative_positive_ratio,
            "strict_leakage_prevention": self.settings.strict_leakage_prevention,
            "dataset": {
                "train_rows": len(y_train),
                "train_rows_before_negative_sampling": original_train_rows,
                "train_positive_rows_before_negative_sampling": original_train_positives,
                "train_positive_rows_after_negative_sampling": int(y_train.eq(1).sum()),
                "train_negative_rows_after_negative_sampling": int(y_train.eq(0).sum()),
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
            "target_audit_status": target_audit["status"],
            "data_drift_status": drift_report["status"],
            "robustness_status": robustness_report["status"],
            "walk_forward_status": walk_forward_report["status"],
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_version,
            "feature_set_version": self.settings.feature_set_version,
            "code_version": current_code_version,
            "experiment_fingerprint": experiment_fingerprint(reproducibility),
            "geo_ablation_enabled": self.settings.run_geo_ablation,
            "feature_exclusions": list(self.settings.feature_exclusions),
            "exclude_geographic_features": self.settings.exclude_geographic_features,
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
            self.settings.artifact_path(self.settings.optuna_trials_filename),
            self.settings.artifact_path(self.settings.optuna_study_filename),
            self.settings.artifact_path(self.settings.target_audit_filename),
            self.settings.artifact_path(self.settings.target_audit_markdown_filename),
            self.settings.artifact_path(self.settings.data_drift_report_filename),
            self.settings.artifact_path(self.settings.data_drift_markdown_filename),
            self.settings.artifact_path(self.settings.feature_stability_report_filename),
            self.settings.artifact_path(self.settings.feature_stability_markdown_filename),
            self.settings.artifact_path(self.settings.robustness_report_filename),
            self.settings.artifact_path(self.settings.geo_ablation_report_filename),
            self.settings.artifact_path(self.settings.threshold_recommendations_filename),
            self.settings.artifact_path(self.settings.error_attribution_report_filename),
        ]
        baseline_decision = BaselineDecisionService(self.settings).decide(
            metadata,
            leakage_report,
            required_for_decision,
            target_audit=target_audit,
            drift_report=drift_report,
            robustness_report=robustness_report,
            walk_forward_report=walk_forward_report,
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
        write_model_review_report(
            self.settings.artifact_path(self.settings.model_review_report_filename),
            metadata,
            leakage_report,
            baseline_decision,
            target_audit,
            drift_report,
            robustness_report,
            walk_forward_report,
            threshold_recommendations,
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
        mlflow_run_id = MlflowTrackingService(self.settings).log_completed_run(
            history_run_dir,
            metadata,
            leakage_report,
            input_example=X_val.head(min(5, len(X_val))),
            prediction_example=validation_scores[: min(5, len(validation_scores))],
        )
        logger.info("Pipeline e metadados salvos em %s", self.settings.artifacts_dir)
        baseline_registry: BaselineRegistry | None = None
        if self.settings.promote_baseline and baseline_decision["decision"] == "approved":
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
        uploaded_artifacts = storage_sync.upload_artifacts(history_run_dir=history_run_dir)
        storage_sync.purge_local_artifacts(uploaded_artifacts)

        return TrainingResult(
            model_name=selected_model_name,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            out_of_time_metrics=out_of_time_metrics,
            pipeline_path=self.settings.pipeline_path,
            metadata_path=self.settings.metadata_path,
            threshold_analysis_path=self.settings.threshold_analysis_path,
            leakage_report_path=self.settings.leakage_report_path,
            run_id=run_id,
            mlflow_run_id=mlflow_run_id,
            history_run_dir=history_run_dir,
            baseline_decision=baseline_decision["decision"],
        )

    def _run_external_benchmarks(
        self,
        pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_out_of_time: pd.DataFrame,
        y_out_of_time: pd.Series,
        run_id: str,
        benchmark_results_path: Path,
        benchmark_summary_path: Path,
    ) -> list[dict[str, Any]]:
        """Run optional external benchmarks without making them a training gate."""
        if not self.settings.external_benchmarks_enabled:
            return self._write_external_benchmark_status(
                benchmark_results_path,
                benchmark_summary_path,
                [{"status": "disabled", "message": "Benchmarks externos desabilitados."}],
            )
        if not self.settings.enabled_external_benchmark_backends:
            return self._write_external_benchmark_status(
                benchmark_results_path,
                benchmark_summary_path,
                [
                    {
                        "status": "disabled",
                        "message": "Nenhum backend de benchmark externo habilitado.",
                    }
                ],
            )

        benchmark_runner = ExternalBenchmarkRunner(self.settings)
        started_at = datetime.now(timezone.utc)
        try:
            benchmark_dataset = benchmark_runner.prepare_dataset(
                pipeline,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                X_out_of_time,
                y_out_of_time,
            )
            return benchmark_runner.run(
                benchmark_dataset,
                results_path=benchmark_results_path,
                summary_path=benchmark_summary_path,
                output_dir=self.settings.artifacts_dir / "external_benchmarks" / run_id,
            )
        except KeyboardInterrupt as exc:
            duration = (datetime.now(timezone.utc) - started_at).total_seconds()
            logger.warning(
                "Etapa de benchmarks externos interrompida; treino principal sera preservado.",
                exc_info=True,
            )
            return self._write_external_benchmark_status(
                benchmark_results_path,
                benchmark_summary_path,
                [
                    {
                        "backend": "external_benchmarks",
                        "status": "interrupted",
                        "message": type(exc).__name__,
                        "duration_seconds": duration,
                    }
                ],
            )
        except Exception as exc:
            duration = (datetime.now(timezone.utc) - started_at).total_seconds()
            logger.warning(
                "Etapa de benchmarks externos falhou; treino principal sera preservado.",
                exc_info=True,
            )
            return self._write_external_benchmark_status(
                benchmark_results_path,
                benchmark_summary_path,
                [
                    {
                        "backend": "external_benchmarks",
                        "status": "failed",
                        "message": str(exc),
                        "duration_seconds": duration,
                    }
                ],
            )

    @staticmethod
    def _write_external_benchmark_status(
        results_path: Path,
        summary_path: Path,
        summary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist empty benchmark results plus a machine-readable status summary."""
        pd.DataFrame(columns=BENCHMARK_RESULT_COLUMNS).to_csv(results_path, index=False)
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return summary

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
