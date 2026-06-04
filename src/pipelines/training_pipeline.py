"""End-to-end training pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config.settings import Settings
from src.data.load_data import RawDataRepository
from src.data.merge_data import FraudDataMerger
from src.data.split_data import TemporalSplitter
from src.features.cleaning import FraudDataCleaner
from src.models.evaluate import evaluate_binary_classifier
from src.models.threshold import find_best_threshold
from src.models.train import FraudModelTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Training output metadata."""

    model_name: str
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    pipeline_path: Path
    metadata_path: Path


class TrainingPipeline:
    """Service layer that executes the full training workflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self) -> TrainingResult:
        """Load data, merge, split, train, tune threshold, evaluate and persist."""
        self.settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

        repository = RawDataRepository(self.settings)
        raw = repository.load_all()
        merged = FraudDataMerger(self.settings).merge(
            transactions=raw["transactions"],
            cards=raw["cards"],
            users=raw["users"],
            mcc_codes=raw["mcc"],
            labels=raw["labels"],
        )

        cleaned_for_split = FraudDataCleaner(self.settings).fit_transform(merged)
        splits = TemporalSplitter(self.settings).split(cleaned_for_split)

        X_train, y_train = self._split_xy(splits.train)
        X_val, y_val = self._split_xy(splits.validation)
        X_test, y_test = self._split_xy(splits.test)

        pipeline = FraudModelTrainer(self.settings).train(X_train, y_train)
        validation_scores = self._predict_scores(pipeline, X_val)
        threshold, threshold_metrics = find_best_threshold(
            y_val.to_numpy(),
            validation_scores,
            beta=self.settings.threshold_beta,
            min_precision=self.settings.min_precision_for_threshold,
        )
        logger.info("Threshold escolhido na validacao: %.4f | %s", threshold, threshold_metrics)

        validation_metrics = evaluate_binary_classifier(
            y_val.to_numpy(), validation_scores, threshold=threshold, beta=self.settings.threshold_beta
        )
        test_scores = self._predict_scores(pipeline, X_test)
        test_metrics = evaluate_binary_classifier(
            y_test.to_numpy(), test_scores, threshold=threshold, beta=self.settings.threshold_beta
        )

        joblib.dump(pipeline, self.settings.pipeline_path)
        metadata = {
            "model_name": self.settings.model_name,
            "threshold": threshold,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "time_column": splits.time_column,
        }
        joblib.dump(metadata, self.settings.metadata_path)
        logger.info("Pipeline e metadados salvos em %s", self.settings.artifacts_dir)

        return TrainingResult(
            model_name=self.settings.model_name,
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            pipeline_path=self.settings.pipeline_path,
            metadata_path=self.settings.metadata_path,
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
