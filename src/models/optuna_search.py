"""Controlled Optuna search over supported sklearn fraud models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.config.settings import Settings
from src.models.evaluate import evaluate_binary_classifier
from src.models.model_factory import ModelFactory
from src.models.temporal_objective import (
    temporal_robustness_score,
    write_objective_breakdown,
)
from src.models.train import FraudModelTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


def temporal_selection_score(
    window_results: list[dict[str, Any]],
    pr_auc_stability_penalty: float,
    recall_stability_penalty: float,
    last_window_penalty: float,
) -> dict[str, float]:
    """Score temporal robustness using worst-window performance over averages."""
    completed = [item for item in window_results if item.get("metrics")]
    if not completed:
        return {
            "selection_score": float("-inf"),
            "min_temporal_pr_auc": 0.0,
            "min_temporal_recall": 0.0,
            "mean_temporal_pr_auc": 0.0,
            "mean_temporal_recall": 0.0,
            "temporal_pr_auc_range": 0.0,
            "temporal_recall_range": 0.0,
            "stability_penalty": 0.0,
            "last_window_penalty": 0.0,
            "last_window_pr_auc": 0.0,
            "last_window_recall": 0.0,
        }
    pr_aucs = np.array([float(item["pr_auc"]) for item in completed], dtype=float)
    recalls = np.array([float(item["metrics"]["recall"]) for item in completed], dtype=float)
    pr_auc_range = float(pr_aucs.max() - pr_aucs.min()) if len(pr_aucs) else 0.0
    recall_range = float(recalls.max() - recalls.min()) if len(recalls) else 0.0
    stability_penalty = (
        pr_auc_stability_penalty * pr_auc_range
        + recall_stability_penalty * recall_range
    )
    last_pr_auc = float(pr_aucs[-1])
    last_recall = float(recalls[-1])
    median_pr_auc = float(np.median(pr_aucs))
    median_recall = float(np.median(recalls))
    last_penalty = last_window_penalty * (
        max(0.0, median_pr_auc - last_pr_auc)
        + max(0.0, median_recall - last_recall)
    )
    selection_score = (
        float(pr_aucs.min())
        + float(recalls.min())
        - stability_penalty
        - last_penalty
    )
    return {
        "selection_score": selection_score,
        "min_temporal_pr_auc": float(pr_aucs.min()),
        "min_temporal_recall": float(recalls.min()),
        "mean_temporal_pr_auc": float(pr_aucs.mean()),
        "mean_temporal_recall": float(recalls.mean()),
        "temporal_pr_auc_range": pr_auc_range,
        "temporal_recall_range": recall_range,
        "stability_penalty": stability_penalty,
        "last_window_penalty": last_penalty,
        "last_window_pr_auc": last_pr_auc,
        "last_window_recall": last_recall,
    }


@dataclass(frozen=True)
class ModelSelectionResult:
    """Selected pipeline and reproducible Optuna study metadata."""

    pipeline: Any
    model_name: str
    model_params: dict[str, Any]
    validation_pr_auc: float
    trial_count: int
    best_trial_breakdown: dict[str, Any]


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
        """Run an independent study per model family and refit its best eligible trial."""
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

        windows = self._selection_windows(X_train, y_train, X_validation, y_validation)
        studies = []
        family_winners = []
        for family_index, model_name in enumerate(available_candidates):
            sampler = optuna.samplers.TPESampler(
                seed=self.settings.random_state + family_index
            )
            pruner = (
                optuna.pruners.MedianPruner()
                if self.settings.optuna_enable_pruning
                else optuna.pruners.NopPruner()
            )
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=self._study_name(model_name),
            )

            def objective(trial, selected_model: str = model_name) -> float:
                params = self._suggest_params(
                    trial,
                    selected_model,
                    positive_class_weight=(
                        self._positive_class_weight(y_train)
                        if self.settings.imbalance_strategy
                        in {"class_weight", "negative_sampling_plus_class_weight"}
                        else 1.0
                    ),
                )
                window_results: list[dict[str, Any]] = []
                for fold_index, window in enumerate(windows):
                    if window["y_train"].nunique() < 2:
                        continue
                    pipeline = FraudModelTrainer(self.settings).train(
                        window["X_train"],
                        window["y_train"],
                        model_name=selected_model,
                        model_params=params,
                    )
                    scores = pipeline.predict_proba(window["X_validation"])[:, 1]
                    metrics = evaluate_binary_classifier(
                        window["y_validation"].to_numpy(),
                        scores,
                        threshold=0.5,
                        beta=self.settings.threshold_beta,
                    )
                    business_cost = (
                        float(metrics["fp"]) * self.settings.false_positive_cost
                        + float(metrics["fn"]) * self.settings.false_negative_cost
                    )
                    metrics["business_cost"] = business_cost
                    metrics["cost_per_record"] = business_cost / max(
                        1, len(window["y_validation"])
                    )
                    metrics["row_count"] = len(window["y_validation"])
                    window_results.append(
                        {
                            "name": window["name"],
                            "pr_auc": metrics["pr_auc"],
                            "metrics": metrics,
                            "evaluation_status": metrics["evaluation_status"],
                            "fraud_rate": float(window["y_validation"].mean()),
                            "row_count": len(window["y_validation"]),
                        }
                    )
                    partial_valid = [
                        item for item in window_results if item["pr_auc"] is not None
                    ]
                    if partial_valid:
                        trial.report(
                            float(np.mean([item["pr_auc"] for item in partial_valid])),
                            step=fold_index,
                        )
                        # Report intermediate quality to the configured pruner. The
                        # complete fold audit is still finished so every persisted
                        # trial has a reproducible score decomposition.

                breakdown = temporal_robustness_score(
                    window_results,
                    min_valid_fold_count=self.settings.min_valid_temporal_folds,
                    min_fold_recall_required=self.settings.min_fold_recall_candidate,
                    min_last_fold_recall_required=self.settings.min_last_fold_recall_candidate,
                    min_pr_auc_lift_required=self.settings.min_pr_auc_lift_over_random,
                    max_alert_rate=self.settings.max_temporal_alert_rate,
                    max_pr_auc_temporal_drop=self.settings.max_pr_auc_temporal_drop,
                    max_recall_temporal_drop=self.settings.max_recall_temporal_drop,
                    false_negative_cost=self.settings.false_negative_cost,
                )
                trial.set_user_attr("model_name", selected_model)
                trial.set_user_attr("evaluation_threshold", 0.5)
                for name, value in breakdown.items():
                    trial.set_user_attr(
                        name,
                        json.dumps(value, sort_keys=True) if isinstance(value, list) else value,
                    )
                trial.set_user_attr(
                    "selection_windows",
                    json.dumps(window_results, sort_keys=True),
                )
                if self.settings.optuna_enable_pruning and trial.should_prune():
                    raise optuna.TrialPruned()
                logger.info(
                    "Optuna trial=%d | model=%s | eligible=%s | raw=%.6f | final=%.6f",
                    trial.number,
                    selected_model,
                    breakdown["eligibility_status"],
                    breakdown["raw_temporal_score"],
                    breakdown["final_objective_score"],
                )
                return float(breakdown["final_objective_score"])

            study.optimize(
                objective,
                n_trials=self.settings.optuna_trials_per_model,
                timeout=(
                    self.settings.optuna_timeout_per_model_seconds
                    or self.settings.optuna_timeout_seconds
                ),
                n_jobs=self.settings.optuna_n_jobs,
                catch=(ImportError, RuntimeError, ValueError, MemoryError),
            )
            studies.append(study)
            eligible_trials = [
                trial
                for trial in study.trials
                if trial.state.name == "COMPLETE"
                and trial.value is not None
                and trial.user_attrs.get("eligibility_status") == "eligible"
            ]
            if eligible_trials:
                winner = max(eligible_trials, key=lambda item: float(item.value))
                family_winners.append((model_name, study, winner))

        rows = self._trial_rows(studies)
        pd.DataFrame(rows).to_csv(trials_path, index=False)
        write_objective_breakdown(
            rows,
            trials_path.with_name(self.settings.objective_score_breakdown_filename),
            trials_path.with_name(
                self.settings.objective_score_breakdown_markdown_filename
            ),
        )
        if not family_winners:
            self._write_study_summary(
                studies, unavailable_candidates, None, study_path
            )
            raise RuntimeError(
                "Nenhum trial elegivel. Consulte objective_score_breakdown.csv."
            )

        best_model_name, best_study, best_trial = max(
            family_winners, key=lambda item: float(item[2].value)
        )
        tuned_params = self._extract_model_params(best_trial.params, best_model_name)
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
        validation_scores = pipeline.predict_proba(X_validation)[:, 1]
        validation_metrics = evaluate_binary_classifier(
            y_validation.to_numpy(), validation_scores, threshold=0.5
        )
        best_validation_pr_auc = float(validation_metrics["pr_auc"] or 0.0)
        self._write_study_summary(
            studies,
            unavailable_candidates,
            {
                "model_name": best_model_name,
                "trial_number": best_trial.number,
                "value": float(best_trial.value),
                "model_params": best_params,
                "validation_pr_auc_after_selection": best_validation_pr_auc,
            },
            study_path,
        )
        trial_count = sum(len(study.trials) for study in studies)
        return ModelSelectionResult(
            pipeline=pipeline,
            model_name=best_model_name,
            model_params=best_params,
            validation_pr_auc=best_validation_pr_auc,
            trial_count=trial_count,
            best_trial_breakdown=dict(best_trial.user_attrs),
        )

    def _selection_windows(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> list[dict[str, Any]]:
        """Build internal expanding windows; external validation/test/OOT stay untouched."""
        windows: list[dict[str, Any]] = []
        if self.settings.optuna_selection_objective in {
            "temporal_stability",
            "temporal_robustness",
        }:
            holdout_size = max(10, int(len(X_train) * self.settings.optuna_temporal_holdout_fraction))
            fold_count = max(
                self.settings.min_valid_temporal_folds,
                self.settings.walk_forward_folds - 1,
            )
            for fold_index in range(fold_count):
                train_end = len(X_train) - holdout_size * (fold_count - fold_index)
                validation_end = train_end + holdout_size
                if train_end <= 0 or validation_end > len(X_train):
                    continue
                fit_X = X_train.iloc[:train_end]
                fit_y = y_train.iloc[:train_end]
                holdout_X = X_train.iloc[train_end:validation_end]
                holdout_y = y_train.iloc[train_end:validation_end]
                if len(fit_X) >= 20 and len(holdout_X) >= 10 and fit_y.nunique() >= 2:
                    windows.append(
                        {
                            "name": f"train_temporal_fold_{len(windows) + 1}",
                            "X_train": fit_X,
                            "y_train": fit_y,
                            "X_validation": holdout_X,
                            "y_validation": holdout_y,
                        }
                    )
        if not windows:
            raise ValueError("Nao foi possivel criar folds temporais internos para o Optuna.")
        return windows

    @staticmethod
    def _study_name(model_name: str) -> str:
        names = {
            "logistic_regression": "fraud_logistic_regression_search",
            "logistic_regression_regularized": "fraud_logistic_regression_search",
            "hist_gradient_boosting": "fraud_hist_gradient_boosting_search",
            "random_forest": "fraud_random_forest_search",
            "random_forest_regularized": "fraud_random_forest_search",
        }
        return names.get(model_name, f"fraud_{model_name}_search")

    def _write_study_summary(
        self,
        studies: list[Any],
        unavailable_candidates: tuple[str, ...],
        winner: dict[str, Any] | None,
        path: Path,
    ) -> None:
        payload = {
            "direction": "maximize",
            "objective": "eligible_temporal_quality_then_stability",
            "optuna_uses_test": False,
            "optuna_uses_out_of_time": False,
            "threshold_uses_test": False,
            "threshold_uses_out_of_time": False,
            "threshold_selected_after_model": True,
            "pruning_enabled": self.settings.optuna_enable_pruning,
            "trials_per_model": self.settings.optuna_trials_per_model,
            "timeout_per_model_seconds": self.settings.optuna_timeout_per_model_seconds,
            "unavailable_candidates": list(unavailable_candidates),
            "eligibility_policy": {
                "min_valid_fold_count": self.settings.min_valid_temporal_folds,
                "min_fold_recall": self.settings.min_fold_recall_candidate,
                "min_last_fold_recall": self.settings.min_last_fold_recall_candidate,
                "min_pr_auc_lift_over_random": self.settings.min_pr_auc_lift_over_random,
                "max_alert_rate": self.settings.max_temporal_alert_rate,
                "max_pr_auc_temporal_drop": self.settings.max_pr_auc_temporal_drop,
                "max_recall_temporal_drop": self.settings.max_recall_temporal_drop,
                "status": "experimental_not_production_policy",
            },
            "studies": [
                {
                    "study_name": study.study_name,
                    "model_name": (
                        study.trials[0].user_attrs.get("model_name")
                        if study.trials
                        else None
                    ),
                    "trial_count": len(study.trials),
                    "eligible_trial_count": sum(
                        trial.user_attrs.get("eligibility_status") == "eligible"
                        for trial in study.trials
                    ),
                }
                for study in studies
            ],
            "winner": winner,
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )

    def _suggest_params(
        self,
        trial,
        model_name: str,
        positive_class_weight: float = 1.0,
    ) -> dict[str, Any]:
        if model_name in {"logistic_regression", "logistic_regression_regularized"}:
            return {
                "C": trial.suggest_float("logistic_regression__C", 1e-3, 100.0, log=True),
                "solver": trial.suggest_categorical(
                    "logistic_regression__solver",
                    ["lbfgs", "liblinear"],
                ),
                "max_iter": 1500,
            }
        if model_name in {
            "random_forest",
            "random_forest_regularized",
            "extra_trees_regularized",
            "balanced_random_forest",
        }:
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
        if model_name in {"xgboost", "xgboost_scale_pos_weight"}:
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
        if model_name in {"lightgbm", "lightgbm_scale_pos_weight"}:
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
        if model_name in {"catboost", "catboost_regularized"}:
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
        if model_name == "easy_ensemble":
            return {
                "n_estimators": trial.suggest_int("easy_ensemble__n_estimators", 10, 40, step=10),
            }
        if model_name == "rus_boost":
            return {
                "n_estimators": trial.suggest_int("rus_boost__n_estimators", 100, 400, step=50),
                "learning_rate": trial.suggest_float(
                    "rus_boost__learning_rate", 0.01, 0.20, log=True
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
        canonical = {
            "logistic_regression_regularized": "logistic_regression",
            "random_forest_regularized": "random_forest",
            "extra_trees_regularized": "random_forest",
            "balanced_random_forest": "random_forest",
            "xgboost_scale_pos_weight": "xgboost",
            "lightgbm_scale_pos_weight": "lightgbm",
            "catboost_regularized": "catboost",
        }.get(model_name, model_name)
        prefix = f"{canonical}__"
        return {
            key.removeprefix(prefix): value
            for key, value in params.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def _trial_rows(studies: list[Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        global_trial_number = 0
        for study in studies:
            for trial in study.trials:
                attrs = trial.user_attrs
                row = {
                    "trial_number": global_trial_number,
                    "family_trial_number": trial.number,
                    "study_name": study.study_name,
                    "state": trial.state.name,
                    "selection_score": trial.value,
                    "model_name": attrs.get("model_name"),
                    "model_params": json.dumps(trial.params, sort_keys=True),
                    "evaluation_threshold": attrs.get("evaluation_threshold"),
                    "selection_windows": attrs.get("selection_windows"),
                    "duration_seconds": (
                        trial.duration.total_seconds() if trial.duration is not None else None
                    ),
                }
                row.update(
                    {
                        key: value
                        for key, value in attrs.items()
                        if key not in {"model_name", "selection_windows"}
                    }
                )
                rows.append(row)
                global_trial_number += 1
        return rows
