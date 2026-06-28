"""Factory and strategies for fraud classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.util import find_spec

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config.settings import Settings


class ModelStrategy(ABC):
    """Strategy interface for creating sklearn classifiers."""

    dependency: str | None = None

    def is_available(self) -> bool:
        """Return whether the strategy runtime dependency is installed."""
        return self.dependency is None or find_spec(self.dependency) is not None

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


class XGBoostStrategy(ModelStrategy):
    """Create an XGBoost classifier without forcing the optional dependency."""

    dependency = "xgboost"

    def create(self, params: dict) -> ClassifierMixin:
        from xgboost import XGBClassifier

        return XGBClassifier(**params)


class LightGBMStrategy(ModelStrategy):
    """Create a LightGBM classifier without forcing the optional dependency."""

    dependency = "lightgbm"

    def create(self, params: dict) -> ClassifierMixin:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params)


class CatBoostStrategy(ModelStrategy):
    """Create a CatBoost classifier without forcing the optional dependency."""

    dependency = "catboost"

    def create(self, params: dict) -> ClassifierMixin:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**params)


class ModelFactory:
    """Factory for supported fraud detection models."""

    _strategies: dict[str, ModelStrategy] = {
        "logistic_regression": LogisticRegressionStrategy(),
        "random_forest": RandomForestStrategy(),
        "hist_gradient_boosting": HistGradientBoostingStrategy(),
        "xgboost": XGBoostStrategy(),
        "lightgbm": LightGBMStrategy(),
        "catboost": CatBoostStrategy(),
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
        strategy = self._strategies[name]
        if not strategy.is_available():
            raise RuntimeError(
                f"Dependencia opcional para {name} nao instalada. "
                "Execute `pip install -r requirements-models.txt`."
            )
        configured_params = {**self.settings.model_params.get(name, {}), **(params or {})}
        return strategy.create(configured_params)

    @classmethod
    def available_model_names(cls, candidates: tuple[str, ...]) -> tuple[str, ...]:
        """Return configured model names whose runtime dependencies are available."""
        return tuple(
            name
            for name in candidates
            if name in cls._strategies and cls._strategies[name].is_available()
        )

    @classmethod
    def unavailable_model_names(cls, candidates: tuple[str, ...]) -> tuple[str, ...]:
        """Return configured model names blocked by an optional dependency."""
        return tuple(
            name
            for name in candidates
            if name in cls._strategies and not cls._strategies[name].is_available()
        )
