"""Factory and strategies for fraud classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config.settings import Settings


class ModelStrategy(ABC):
    """Strategy interface for creating sklearn classifiers."""

    @abstractmethod
    def create(self, params: dict) -> ClassifierMixin:
        """Create a classifier instance."""


class LogisticRegressionStrategy(ModelStrategy):
    """Create a balanced logistic regression baseline."""

    def create(self, params: dict) -> ClassifierMixin:
        return LogisticRegression(**params)


class RandomForestStrategy(ModelStrategy):
    """Create a random forest classifier."""

    def create(self, params: dict) -> ClassifierMixin:
        return RandomForestClassifier(**params)


class HistGradientBoostingStrategy(ModelStrategy):
    """Create a histogram gradient boosting classifier."""

    def create(self, params: dict) -> ClassifierMixin:
        return HistGradientBoostingClassifier(**params)


class ModelFactory:
    """Factory for supported fraud detection models."""

    _strategies: dict[str, ModelStrategy] = {
        "logistic_regression": LogisticRegressionStrategy(),
        "random_forest": RandomForestStrategy(),
        "hist_gradient_boosting": HistGradientBoostingStrategy(),
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(
        self,
        model_name: str | None = None,
        params: dict | None = None,
    ) -> ClassifierMixin:
        """Create a model by name."""
        name = model_name or self.settings.model_name
        if name not in self._strategies:
            supported = ", ".join(sorted(self._strategies))
            raise ValueError(f"Modelo nao suportado: {name}. Suportados: {supported}")
        configured_params = {**self.settings.model_params.get(name, {}), **(params or {})}
        return self._strategies[name].create(configured_params)
