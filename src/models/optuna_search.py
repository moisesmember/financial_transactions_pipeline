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
from src.models.model_factory import ModelFactory
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
    """Select model family and hyperparameters with temporal stability controls."""

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

        configured_candidates = self.settings.optuna_model_candidates
        available_candidates = ModelFactory.available_model_names(configured_candidates)
        unavailable_candidates = ModelFactory.unavailable_model_names(configured_candidates)
        if unavailable_candidates:
            logger.warning(
                "Modelos Optuna ignorados por dependencia ausente: %s. "
                "Instale com `pip install -r requirements-models.txt`.",
                ", ".join(unavailable_candidates),
            )
        if not available_candidates:
            raise RuntimeError(
                "Nenhum candidato Optuna possui as dependencias instaladas. "
                "Execute `pip install -r requirements-models.txt` ou configure "
                "um modelo nativo do scikit-learn."
            )

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
        for model_name in available_candidates:
            study.enqueue_trial({"model_name": model_name})

        def objective(trial) -> float:
            model_name = trial.suggest_categorical(
                "model_name",
                list(available_candidates),
            )
            params = self._suggest_params(
                trial,
                model_name,
                positive_class_weight=self._positive_class_weight(y_train),
            )
            windows = self._selection_windows(X_train, y_train, X_validation, y_validation)
            window_results = []
            for window in windows:
                pipeline = FraudModelTrainer(self.settings).train(
                    window["X_train"],
                    window["y_train"],
                    model_name=model_name,
                    model_params=params,
                )
                scores = pipeline.predict_proba(window["X_validation"])[:, 1]
                pr_auc = float(
                    average_precision_score(window["y_validation"].to_numpy(), scores)
                )
                table = build_threshold_table(
                    window["y_validation"].to_numpy(),
                    scores,
                    thresholds=thresholds,
                    beta=self.settings.threshold_beta,
                    false_positive_cost=self.settings.false_positive_cost,
                    false_negative_cost=self.settings.false_negative_cost,
                    split="validation",
                )
                selected_threshold, metrics = select_business_threshold(table)
                metrics["alert_rate"] = float(
                    (metrics["tp"] + metrics["fp"])
                    / max(1.0, metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"])
                )
                window_results.append(
                    {
                        "name": window["name"],
                        "pr_auc": pr_auc,
                        "threshold": selected_threshold,
                        "metrics": metrics,
                    }
                )
            validation_result = window_results[-1]
            pr_aucs = np.array([item["pr_auc"] for item in window_results], dtype=float)
            pr_auc_range = float(pr_aucs.max() - pr_aucs.min()) if len(pr_aucs) else 0.0
            mean_pr_auc = float(pr_aucs.mean()) if len(pr_aucs) else 0.0
            stability_penalty = self.settings.optuna_pr_auc_stability_penalty * pr_auc_range
            if self.settings.optuna_selection_objective == "temporal_stability":
                objective_value = mean_pr_auc - stability_penalty
            else:
                objective_value = validation_result["pr_auc"]
            selected_threshold = validation_result["threshold"]
            metrics = validation_result["metrics"]
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
            trial.set_user_attr("validation_pr_auc", validation_result["pr_auc"])
            trial.set_user_attr("mean_temporal_pr_auc", mean_pr_auc)
            trial.set_user_attr("min_temporal_pr_auc", float(pr_aucs.min()))
            trial.set_user_attr("max_temporal_pr_auc", float(pr_aucs.max()))
            trial.set_user_attr("temporal_pr_auc_range", pr_auc_range)
            trial.set_user_attr("stability_penalty", stability_penalty)
            trial.set_user_attr(
                "selection_windows",
                json.dumps(
                    [
                        {
                            "name": item["name"],
                            "pr_auc": item["pr_auc"],
                            "threshold": item["threshold"],
                            "recall": item["metrics"]["recall"],
                            "alert_rate": item["metrics"]["alert_rate"],
                            "business_cost": item["metrics"]["business_cost"],
                        }
                        for item in window_results
                    ],
                    sort_keys=True,
                ),
            )
            logger.info(
                "Optuna trial=%d | model=%s | objective=%s | score=%.6f | validation_pr_auc=%.6f | temporal_range=%.6f | threshold=%.4f",
                trial.number,
                model_name,
                self.settings.optuna_selection_objective,
                objective_value,
                validation_result["pr_auc"],
                pr_auc_range,
                selected_threshold,
            )
            return objective_value

        study.optimize(
            objective,
            n_trials=max(self.settings.optuna_trials, len(available_candidates)),
            timeout=self.settings.optuna_timeout_seconds,
            n_jobs=self.settings.optuna_n_jobs,
            catch=(ImportError, RuntimeError, ValueError, MemoryError),
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
        best_validation_pr_auc = float(
            study.best_trial.user_attrs.get("validation_pr_auc", study.best_value)
        )
        self._write_trials(study, trials_path)
        study_path.write_text(
            json.dumps(
                {
                    "study_name": study.study_name,
                    "direction": "maximize",
                    "objective": self.settings.optuna_selection_objective,
                    "sampler": "TPESampler",
                    "seed": self.settings.random_state,
                    "temporal_holdout_fraction": self.settings.optuna_temporal_holdout_fraction,
                    "pr_auc_stability_penalty": self.settings.optuna_pr_auc_stability_penalty,
                    "configured_candidates": list(configured_candidates),
                    "available_candidates": list(available_candidates),
                    "unavailable_candidates": list(unavailable_candidates),
                    "best_trial_number": study.best_trial.number,
                    "best_value": float(study.best_value),
                    "best_validation_pr_auc": best_validation_pr_auc,
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
            validation_pr_auc=best_validation_pr_auc,
            trial_count=len(study.trials),
        )

    def _selection_windows(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> list[dict[str, Any]]:
        """Build temporal windows used for model selection without touching test/OOT."""
        windows: list[dict[str, Any]] = []
        if self.settings.optuna_selection_objective == "temporal_stability":
            holdout_start = int(len(X_train) * (1 - self.settings.optuna_temporal_holdout_fraction))
            fit_X = X_train.iloc[:holdout_start]
            fit_y = y_train.iloc[:holdout_start]
            holdout_X = X_train.iloc[holdout_start:]
            holdout_y = y_train.iloc[holdout_start:]
            if (
                len(fit_X) >= 20
                and len(holdout_X) >= 10
                and fit_y.nunique() >= 2
                and holdout_y.nunique() >= 2
            ):
                windows.append(
                    {
                        "name": "train_tail_holdout",
                        "X_train": fit_X,
                        "y_train": fit_y,
                        "X_validation": holdout_X,
                        "y_validation": holdout_y,
                    }
                )
        windows.append(
            {
                "name": "validation",
                "X_train": X_train,
                "y_train": y_train,
                "X_validation": X_validation,
                "y_validation": y_validation,
            }
        )
        return windows

    def _suggest_params(
        self,
        trial,
        model_name: str,
        positive_class_weight: float = 1.0,
    ) -> dict[str, Any]:
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
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int(
                    "xgboost__n_estimators",
                    150,
                    500,
                    step=50,
                ),
                "max_depth": trial.suggest_int("xgboost__max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "xgboost__learning_rate",
                    0.01,
                    0.20,
                    log=True,
                ),
                "subsample": trial.suggest_float("xgboost__subsample", 0.60, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "xgboost__colsample_bytree",
                    0.60,
                    1.0,
                ),
                "min_child_weight": trial.suggest_float(
                    "xgboost__min_child_weight",
                    1.0,
                    20.0,
                    log=True,
                ),
                "reg_alpha": trial.suggest_float(
                    "xgboost__reg_alpha",
                    1e-8,
                    10.0,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "xgboost__reg_lambda",
                    1e-3,
                    20.0,
                    log=True,
                ),
                "scale_pos_weight": trial.suggest_categorical(
                    "xgboost__scale_pos_weight",
                    sorted({1.0, float(np.sqrt(positive_class_weight)), positive_class_weight}),
                ),
            }
        if model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int(
                    "lightgbm__n_estimators",
                    150,
                    500,
                    step=50,
                ),
                "num_leaves": trial.suggest_int("lightgbm__num_leaves", 15, 127),
                "max_depth": trial.suggest_int("lightgbm__max_depth", 4, 12),
                "learning_rate": trial.suggest_float(
                    "lightgbm__learning_rate",
                    0.01,
                    0.20,
                    log=True,
                ),
                "min_child_samples": trial.suggest_int(
                    "lightgbm__min_child_samples",
                    10,
                    100,
                ),
                "subsample": trial.suggest_float("lightgbm__subsample", 0.60, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float(
                    "lightgbm__colsample_bytree",
                    0.60,
                    1.0,
                ),
                "reg_alpha": trial.suggest_float(
                    "lightgbm__reg_alpha",
                    1e-8,
                    10.0,
                    log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "lightgbm__reg_lambda",
                    1e-3,
                    20.0,
                    log=True,
                ),
            }
        if model_name == "catboost":
            return {
                "iterations": trial.suggest_int(
                    "catboost__iterations",
                    150,
                    500,
                    step=50,
                ),
                "depth": trial.suggest_int("catboost__depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "catboost__learning_rate",
                    0.01,
                    0.20,
                    log=True,
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "catboost__l2_leaf_reg",
                    0.1,
                    20.0,
                    log=True,
                ),
                "random_strength": trial.suggest_float(
                    "catboost__random_strength",
                    1e-3,
                    10.0,
                    log=True,
                ),
                "border_count": trial.suggest_int(
                    "catboost__border_count",
                    32,
                    128,
                    step=16,
                ),
            }
        raise ValueError(f"Modelo Optuna nao suportado: {model_name}")

    @staticmethod
    def _positive_class_weight(target: pd.Series) -> float:
        """Return the negative-to-positive ratio used by XGBoost candidates."""
        positive_count = int((target == 1).sum())
        negative_count = int((target == 0).sum())
        if positive_count == 0:
            return 1.0
        return max(1.0, negative_count / positive_count)

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
                    "selection_score": trial.value,
                    "validation_pr_auc": trial.user_attrs.get("validation_pr_auc", trial.value),
                    "mean_temporal_pr_auc": trial.user_attrs.get("mean_temporal_pr_auc"),
                    "min_temporal_pr_auc": trial.user_attrs.get("min_temporal_pr_auc"),
                    "max_temporal_pr_auc": trial.user_attrs.get("max_temporal_pr_auc"),
                    "temporal_pr_auc_range": trial.user_attrs.get("temporal_pr_auc_range"),
                    "stability_penalty": trial.user_attrs.get("stability_penalty"),
                    "selection_windows": trial.user_attrs.get("selection_windows"),
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
