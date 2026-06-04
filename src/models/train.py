"""Model training service."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config.settings import Settings
from src.features.cleaning import FraudDataCleaner
from src.features.feature_engineering import FraudFeatureEngineer
from src.features.preprocessing import build_preprocessor
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger


logger = get_logger(__name__)


class FraudModelTrainer:
    """Train a complete sklearn pipeline for fraud detection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """Create cleaning, feature, preprocessing and model pipeline."""
        cleaner = FraudDataCleaner(self.settings)
        feature_engineer = FraudFeatureEngineer(self.settings)
        sample = cleaner.fit_transform(X_train)
        sample = feature_engineer.fit_transform(sample)
        preprocessor = build_preprocessor(sample, self.settings)
        model = ModelFactory(self.settings).create()

        return Pipeline(
            steps=[
                ("cleaner", FraudDataCleaner(self.settings)),
                ("features", FraudFeatureEngineer(self.settings)),
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Fit and return the complete fraud detection pipeline."""
        pipeline = self.build_pipeline(X_train)
        logger.info("Treinando modelo %s com %s linhas", self.settings.model_name, len(X_train))
        pipeline.fit(X_train, y_train)
        return pipeline
