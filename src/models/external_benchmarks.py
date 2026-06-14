"""Optional external AutoML benchmarks with common temporal evaluation."""

from __future__ import annotations

import importlib.util
import json
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.features.preprocessing import columns_to_drop
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Feature frames shared by external benchmark adapters."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    X_out_of_time: pd.DataFrame
    y_out_of_time: pd.Series


@dataclass(frozen=True)
class BenchmarkFitResult:
    """Adapter-specific fitted predictor and descriptive metadata."""

    predictor: Any
    model_name: str
    metadata: dict[str, Any]


class ExternalBenchmarkAdapter(ABC):
    """Contract implemented by optional external AutoML frameworks."""

    name: str
    import_name: str

    def is_available(self) -> bool:
        return importlib.util.find_spec(self.import_name) is not None

    @abstractmethod
    def fit(self, dataset: BenchmarkDataset, settings: Settings, output_dir: Path) -> BenchmarkFitResult:
        """Fit the benchmark using training and validation only."""

    @abstractmethod
    def predict_scores(self, fitted: BenchmarkFitResult, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class scores."""


class AutoGluonBenchmarkAdapter(ExternalBenchmarkAdapter):
    name = "autogluon"
    import_name = "autogluon"

    def fit(self, dataset: BenchmarkDataset, settings: Settings, output_dir: Path) -> BenchmarkFitResult:
        from autogluon.tabular import TabularPredictor

        label = settings.target_column
        train = dataset.X_train.copy()
        train[label] = dataset.y_train.to_numpy()
        validation = dataset.X_validation.copy()
        validation[label] = dataset.y_validation.to_numpy()
        path = output_dir / "autogluon"
        predictor = TabularPredictor(
            label=label,
            problem_type="binary",
            eval_metric="average_precision",
            path=str(path),
            verbosity=0,
        ).fit(
            train_data=train,
            tuning_data=validation,
            time_limit=settings.external_benchmark_time_limit_seconds,
            presets="medium_quality",
        )
        return BenchmarkFitResult(
            predictor=predictor,
            model_name=str(predictor.model_best),
            metadata={"path": str(path)},
        )

    def predict_scores(self, fitted: BenchmarkFitResult, X: pd.DataFrame) -> np.ndarray:
        probabilities = fitted.predictor.predict_proba(X)
        if isinstance(probabilities, pd.DataFrame):
            return probabilities.iloc[:, -1].to_numpy(dtype=float)
        return np.asarray(probabilities, dtype=float)


class H2OBenchmarkAdapter(ExternalBenchmarkAdapter):
    name = "h2o"
    import_name = "h2o"

    def fit(self, dataset: BenchmarkDataset, settings: Settings, output_dir: Path) -> BenchmarkFitResult:
        import h2o
        from h2o.automl import H2OAutoML

        h2o.init()
        label = settings.target_column
        train = dataset.X_train.copy()
        train[label] = dataset.y_train.astype(str).to_numpy()
        validation = dataset.X_validation.copy()
        validation[label] = dataset.y_validation.astype(str).to_numpy()
        training_frame = h2o.H2OFrame(train)
        validation_frame = h2o.H2OFrame(validation)
        training_frame[label] = training_frame[label].asfactor()
        validation_frame[label] = validation_frame[label].asfactor()
        automl = H2OAutoML(
            max_models=settings.external_benchmark_max_models,
            max_runtime_secs=settings.external_benchmark_time_limit_seconds,
            nfolds=0,
            seed=settings.random_state,
            stopping_metric="AUCPR",
            sort_metric="AUCPR",
            exclude_algos=["DeepLearning"],
        )
        automl.train(
            y=label,
            training_frame=training_frame,
            validation_frame=validation_frame,
            leaderboard_frame=validation_frame,
        )
        return BenchmarkFitResult(
            predictor=automl.leader,
            model_name=str(automl.leader.model_id),
            metadata={
                "leaderboard": h2o.as_list(automl.leaderboard).to_dict(orient="records"),
            },
        )

    def predict_scores(self, fitted: BenchmarkFitResult, X: pd.DataFrame) -> np.ndarray:
        import h2o

        predictions = h2o.as_list(fitted.predictor.predict(h2o.H2OFrame(X)))
        probability_column = "p1" if "p1" in predictions.columns else predictions.columns[-1]
        return predictions[probability_column].to_numpy(dtype=float)


class FlamlBenchmarkAdapter(ExternalBenchmarkAdapter):
    name = "flaml"
    import_name = "flaml"

    def fit(self, dataset: BenchmarkDataset, settings: Settings, output_dir: Path) -> BenchmarkFitResult:
        from flaml import AutoML

        predictor = AutoML()
        predictor.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_validation,
            y_val=dataset.y_validation,
            task="classification",
            metric="ap",
            eval_method="holdout",
            time_budget=settings.external_benchmark_time_limit_seconds,
            seed=settings.random_state,
            n_jobs=1,
            verbose=0,
            log_file_name=str(output_dir / "flaml.log"),
        )
        return BenchmarkFitResult(
            predictor=predictor,
            model_name=str(predictor.best_estimator),
            metadata={
                "best_config": predictor.best_config,
                "best_loss": predictor.best_loss,
            },
        )

    def predict_scores(self, fitted: BenchmarkFitResult, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(fitted.predictor.predict_proba(X), dtype=float)[:, 1]


class ExternalBenchmarkRunner:
    """Execute external frameworks under the same split and threshold protocol."""

    _adapters: dict[str, type[ExternalBenchmarkAdapter]] = {
        "autogluon": AutoGluonBenchmarkAdapter,
        "h2o": H2OBenchmarkAdapter,
        "flaml": FlamlBenchmarkAdapter,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def prepare_dataset(
        self,
        fitted_pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_out_of_time: pd.DataFrame,
        y_out_of_time: pd.Series,
    ) -> BenchmarkDataset:
        """Apply the governed cleaner/features and leave encoding to each framework."""
        cleaner = deepcopy(fitted_pipeline.named_steps["cleaner"])
        features = deepcopy(fitted_pipeline.named_steps["features"])

        def transform(frame: pd.DataFrame, training: bool = False) -> pd.DataFrame:
            cleaned = cleaner.transform(frame)
            engineered = (
                features.fit_transform(cleaned)
                if training
                else features.transform(cleaned)
            )
            drop_columns = columns_to_drop(engineered.columns, self.settings)
            output = engineered.drop(columns=drop_columns, errors="ignore").copy()
            for column in output.select_dtypes(include=["object"]).columns:
                output[column] = output[column].fillna("__missing__").astype("category")
            return output.reset_index(drop=True)

        return BenchmarkDataset(
            X_train=transform(X_train, training=True),
            y_train=y_train.reset_index(drop=True),
            X_validation=transform(X_validation).reset_index(drop=True),
            y_validation=y_validation.reset_index(drop=True),
            X_test=transform(X_test).reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            X_out_of_time=transform(X_out_of_time).reset_index(drop=True),
            y_out_of_time=y_out_of_time.reset_index(drop=True),
        )

    def run(
        self,
        dataset: BenchmarkDataset,
        results_path: Path,
        summary_path: Path,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Run configured adapters and persist metrics plus execution status."""
        output_dir.mkdir(parents=True, exist_ok=True)
        thresholds = threshold_grid(
            self.settings.threshold_analysis_start,
            self.settings.threshold_analysis_stop,
            self.settings.threshold_analysis_step,
        )
        rows: list[dict[str, Any]] = []
        summary: list[dict[str, Any]] = []
        for backend in self.settings.external_benchmark_backends:
            adapter = self._adapters[backend]()
            if not adapter.is_available():
                summary.append(
                    {
                        "backend": backend,
                        "status": "unavailable",
                        "message": f"Dependencia opcional `{adapter.import_name}` nao instalada.",
                    }
                )
                continue
            started = perf_counter()
            try:
                fitted = adapter.fit(dataset, self.settings, output_dir)
                validation_scores = adapter.predict_scores(fitted, dataset.X_validation)
                validation_table = build_threshold_table(
                    dataset.y_validation.to_numpy(),
                    validation_scores,
                    thresholds=thresholds,
                    beta=self.settings.threshold_beta,
                    false_positive_cost=self.settings.false_positive_cost,
                    false_negative_cost=self.settings.false_negative_cost,
                    split="validation",
                )
                selected_threshold, _ = select_business_threshold(validation_table)
                split_payloads = {
                    "validation": (dataset.y_validation, validation_scores),
                    "test": (
                        dataset.y_test,
                        adapter.predict_scores(fitted, dataset.X_test),
                    ),
                    "out_of_time": (
                        dataset.y_out_of_time,
                        adapter.predict_scores(fitted, dataset.X_out_of_time),
                    ),
                }
                for split, (target, scores) in split_payloads.items():
                    metrics = evaluate_binary_classifier(
                        target.to_numpy(),
                        scores,
                        threshold=selected_threshold,
                        beta=self.settings.threshold_beta,
                    )
                    rows.append(
                        {
                            "backend": backend,
                            "framework_model": fitted.model_name,
                            "split": split,
                            **metrics,
                            "business_cost": (
                                metrics["fp"] * self.settings.false_positive_cost
                                + metrics["fn"] * self.settings.false_negative_cost
                            ),
                        }
                    )
                summary.append(
                    {
                        "backend": backend,
                        "status": "completed",
                        "framework_model": fitted.model_name,
                        "duration_seconds": perf_counter() - started,
                        "metadata": fitted.metadata,
                    }
                )
            except Exception as exc:
                logger.exception("Benchmark externo falhou | backend=%s", backend)
                summary.append(
                    {
                        "backend": backend,
                        "status": "failed",
                        "message": str(exc),
                        "duration_seconds": perf_counter() - started,
                    }
                )
                if self.settings.external_benchmark_fail_fast:
                    raise

        pd.DataFrame(rows).to_csv(results_path, index=False)
        safe_summary = _json_safe(summary)
        summary_path.write_text(
            json.dumps(
                safe_summary,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return safe_summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
