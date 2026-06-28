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

    def build_pipeline(
        self,
        X_train: pd.DataFrame,
        model_name: str | None = None,
        model_params: dict | None = None,
    ) -> Pipeline:
        """Create cleaning, feature, preprocessing and model pipeline."""
        cleaner = FraudDataCleaner(self.settings)
        feature_engineer = FraudFeatureEngineer(self.settings)
        sample = cleaner.fit_transform(X_train)
        sample = feature_engineer.fit_transform(sample)
        selected_model = model_name or self.settings.model_name
        preprocessor = build_preprocessor(sample, self.settings, model_name=selected_model)
        model = ModelFactory(self.settings).create(model_name=model_name, params=model_params)

        return Pipeline(
            steps=[
                ("cleaner", FraudDataCleaner(self.settings)),
                ("features", FraudFeatureEngineer(self.settings)),
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str | None = None,
        model_params: dict | None = None,
    ) -> Pipeline:
        """Fit and return the complete fraud detection pipeline."""
        selected_model = model_name or self.settings.model_name
        pipeline = self.build_pipeline(
            X_train,
            model_name=selected_model,
            model_params=model_params,
        )
        logger.info("Treinando modelo %s com %s linhas", selected_model, len(X_train))
        pipeline.fit(X_train, y_train)
        return pipeline
