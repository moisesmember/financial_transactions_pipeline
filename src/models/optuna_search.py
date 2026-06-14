"""Controlled Optuna search over supported sklearn fraud models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config.settings import Settings
from src.models.threshold_analysis import build_threshold_table, select_business_threshold, threshold_grid
from src.models.train import FraudModelTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSelectionResult:
    """Selected pipeline and reproducible Optuna study metadata."""

    pipeline: Any
    model_name: str
    model_params: dict[str, Any]
    validation_pr_auc: float
    trial_count: int


class OptunaModelSelector:
    """Select model family and hyperparameters using validation PR-AUC only."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
        trials_path: Path,
        study_path: Path,
    ) -> ModelSelectionResult:
        """Run seeded TPE search and refit the best configuration on training data."""
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna nao esta instalado. Execute `pip install -r requirements.txt`."
            ) from exc

        thresholds = threshold_grid(
            self.settings.threshold_analysis_start,
            self.settings.threshold_analysis_stop,
            self.settings.threshold_analysis_step,
        )
        sampler = optuna.samplers.TPESampler(seed=self.settings.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="fraud_model_selection",
        )
        for model_name in self.settings.optuna_model_candidates:
            study.enqueue_trial({"model_name": model_name})

        def objective(trial) -> float:
            model_name = trial.suggest_categorical(
                "model_name",
                list(self.settings.optuna_model_candidates),
            )
            params = self._suggest_params(trial, model_name)
            pipeline = FraudModelTrainer(self.settings).train(
                X_train,
                y_train,
                model_name=model_name,
                model_params=params,
            )
            validation_scores = pipeline.predict_proba(X_validation)[:, 1]
            pr_auc = float(
                average_precision_score(y_validation.to_numpy(), validation_scores)
            )
            table = build_threshold_table(
                y_validation.to_numpy(),
                validation_scores,
                thresholds=thresholds,
                beta=self.settings.threshold_beta,
                false_positive_cost=self.settings.false_positive_cost,
                false_negative_cost=self.settings.false_negative_cost,
                split="validation",
            )
            selected_threshold, metrics = select_business_threshold(table)
            trial.set_user_attr("selected_threshold", selected_threshold)
            trial.set_user_attr("precision", metrics["precision"])
            trial.set_user_attr("recall", metrics["recall"])
            trial.set_user_attr(
                "alert_rate",
                float(
                    (metrics["tp"] + metrics["fp"])
                    / max(1, metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"])
                ),
            )
            trial.set_user_attr("business_cost", metrics["business_cost"])
            logger.info(
                "Optuna trial=%d | model=%s | validation_pr_auc=%.6f | threshold=%.4f",
                trial.number,
                model_name,
                pr_auc,
                selected_threshold,
            )
            return pr_auc

        study.optimize(
            objective,
            n_trials=max(self.settings.optuna_trials, len(self.settings.optuna_model_candidates)),
            timeout=self.settings.optuna_timeout_seconds,
            n_jobs=self.settings.optuna_n_jobs,
            catch=(ValueError, MemoryError),
        )
        if study.best_trial.value is None:
            raise RuntimeError("Optuna nao concluiu nenhum trial valido.")

        best_model_name = str(study.best_trial.params["model_name"])
        tuned_params = self._extract_model_params(study.best_trial.params, best_model_name)
        best_params = {
            **self.settings.model_params[best_model_name],
            **tuned_params,
        }
        pipeline = FraudModelTrainer(self.settings).train(
            X_train,
            y_train,
            model_name=best_model_name,
            model_params=best_params,
        )
        self._write_trials(study, trials_path)
        study_path.write_text(
            json.dumps(
                {
                    "study_name": study.study_name,
                    "direction": "maximize",
                    "objective": "validation_pr_auc",
                    "sampler": "TPESampler",
                    "seed": self.settings.random_state,
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "best_model_name": best_model_name,
                    "best_model_params": best_params,
                    "trial_count": len(study.trials),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        return ModelSelectionResult(
            pipeline=pipeline,
            model_name=best_model_name,
            model_params=best_params,
            validation_pr_auc=float(study.best_value),
            trial_count=len(study.trials),
        )

    def _suggest_params(self, trial, model_name: str) -> dict[str, Any]:
        if model_name == "logistic_regression":
            return {
                "C": trial.suggest_float("logistic_regression__C", 1e-3, 100.0, log=True),
                "solver": trial.suggest_categorical(
                    "logistic_regression__solver",
                    ["lbfgs", "liblinear"],
                ),
                "max_iter": 1500,
            }
        if model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int(
                    "random_forest__n_estimators",
                    100,
                    500,
                    step=50,
                ),
                "max_depth": trial.suggest_int("random_forest__max_depth", 6, 24),
                "min_samples_leaf": trial.suggest_int(
                    "random_forest__min_samples_leaf",
                    2,
                    30,
                ),
                "max_features": trial.suggest_categorical(
                    "random_forest__max_features",
                    ["sqrt", "log2", 0.5],
                ),
            }
        if model_name == "hist_gradient_boosting":
            return {
                "learning_rate": trial.suggest_float(
                    "hist_gradient_boosting__learning_rate",
                    0.01,
                    0.20,
                    log=True,
                ),
                "max_iter": trial.suggest_int(
                    "hist_gradient_boosting__max_iter",
                    100,
                    400,
                    step=50,
                ),
                "max_leaf_nodes": trial.suggest_int(
                    "hist_gradient_boosting__max_leaf_nodes",
                    15,
                    63,
                ),
                "min_samples_leaf": trial.suggest_int(
                    "hist_gradient_boosting__min_samples_leaf",
                    10,
                    100,
                ),
                "l2_regularization": trial.suggest_float(
                    "hist_gradient_boosting__l2_regularization",
                    1e-4,
                    10.0,
                    log=True,
                ),
            }
        raise ValueError(f"Modelo Optuna nao suportado: {model_name}")

    @staticmethod
    def _extract_model_params(
        params: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        prefix = f"{model_name}__"
        return {
            key.removeprefix(prefix): value
            for key, value in params.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _write_trials(study, path: Path) -> None:
        rows = []
        for trial in study.trials:
            rows.append(
                {
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "validation_pr_auc": trial.value,
                    "model_name": trial.params.get("model_name"),
                    "model_params": json.dumps(trial.params, sort_keys=True),
                    "selected_threshold": trial.user_attrs.get("selected_threshold"),
                    "precision": trial.user_attrs.get("precision"),
                    "recall": trial.user_attrs.get("recall"),
                    "alert_rate": trial.user_attrs.get("alert_rate"),
                    "business_cost": trial.user_attrs.get("business_cost"),
                    "duration_seconds": (
                        trial.duration.total_seconds() if trial.duration is not None else None
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)
