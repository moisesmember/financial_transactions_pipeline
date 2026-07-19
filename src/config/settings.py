"""Application settings for the fraud detection project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps local imports working before dependency install
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool_alias(name: str, legacy_name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with a backward-compatible alias."""
    if os.getenv(name) is not None:
        return _env_bool(name, default)
    if os.getenv(legacy_name) is not None:
        return _env_bool(legacy_name, default)
    return default


def _env_optional_positive_int(name: str, default: int | None = None) -> int | None:
    """Parse a positive integer, treating empty or zero as unlimited."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    if not normalized or normalized == "0":
        return None
    parsed = int(normalized)
    if parsed < 0:
        raise ValueError(f"{name} deve ser um inteiro positivo ou 0.")
    return parsed


def _env_nonnegative_int(name: str, default: int = 0) -> int:
    """Parse an integer where zero explicitly means unlimited."""
    value = os.getenv(name)
    parsed = default if value is None or not value.strip() else int(value.strip())
    if parsed < 0:
        raise ValueError(f"{name} deve ser um inteiro positivo ou 0.")
    return parsed


def _env_float(name: str, default: float) -> float:
    """Parse a floating-point environment variable."""
    value = os.getenv(name)
    return default if value is None else float(value.strip())


def _env_cost_scenarios(
    name: str,
    default: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Parse comma-separated FP:FN business-cost scenarios."""
    value = os.getenv(name)
    if value is None:
        return default
    scenarios: list[tuple[float, float]] = []
    for item in value.split(","):
        false_positive_cost, false_negative_cost = item.strip().split(":", maxsplit=1)
        scenarios.append((float(false_positive_cost), float(false_negative_cost)))
    return tuple(scenarios)


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Centralized configuration for paths, data columns and modeling."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path | None = None
    artifacts_dir: Path | None = None
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "logistic_regression"))
    model_selection_engine: str = field(
        default_factory=lambda: os.getenv("MODEL_SELECTION_ENGINE", "optuna")
    )
    optuna_model_candidates: tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "OPTUNA_MODEL_CANDIDATES",
            ("logistic_regression", "random_forest", "hist_gradient_boosting"),
        )
    )
    optuna_trials: int = field(default_factory=lambda: int(os.getenv("OPTUNA_TRIALS", "15")))
    optuna_trials_per_model: int = field(
        default_factory=lambda: int(os.getenv("OPTUNA_TRIALS_PER_MODEL", "10"))
    )
    optuna_timeout_seconds: int | None = field(
        default_factory=lambda: _env_optional_positive_int("OPTUNA_TIMEOUT_SECONDS", 900)
    )
    optuna_n_jobs: int = field(default_factory=lambda: int(os.getenv("OPTUNA_N_JOBS", "1")))
    optuna_timeout_per_model_seconds: int | None = field(
        default_factory=lambda: _env_optional_positive_int(
            "OPTUNA_TIMEOUT_PER_MODEL_SECONDS", 3600
        )
    )
    optuna_enable_pruning: bool = field(
        default_factory=lambda: _env_bool("OPTUNA_ENABLE_PRUNING", True)
    )
    optuna_selection_objective: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_SELECTION_OBJECTIVE",
            os.getenv("OPTUNA_SELECTION_OBJECTIVE", "temporal_stability"),
        )
    )
    optuna_temporal_holdout_fraction: float = field(
        default_factory=lambda: _env_float("OPTUNA_TEMPORAL_HOLDOUT_FRACTION", 0.20)
    )
    optuna_pr_auc_stability_penalty: float = field(
        default_factory=lambda: _env_float("OPTUNA_PR_AUC_STABILITY_PENALTY", 0.50)
    )
    optuna_recall_stability_penalty: float = field(
        default_factory=lambda: _env_float("OPTUNA_RECALL_STABILITY_PENALTY", 0.50)
    )
    optuna_last_window_penalty: float = field(
        default_factory=lambda: _env_float("OPTUNA_LAST_WINDOW_PENALTY", 1.00)
    )
    min_valid_temporal_folds: int = field(
        default_factory=lambda: int(os.getenv("MIN_VALID_TEMPORAL_FOLDS", "3"))
    )
    min_fold_recall_candidate: float = field(
        default_factory=lambda: _env_float("MIN_FOLD_RECALL_CANDIDATE", 0.05)
    )
    min_last_fold_recall_candidate: float = field(
        default_factory=lambda: _env_float("MIN_LAST_FOLD_RECALL_CANDIDATE", 0.05)
    )
    min_pr_auc_lift_over_random: float = field(
        default_factory=lambda: _env_float("MIN_PR_AUC_LIFT_OVER_RANDOM", 1.25)
    )
    max_temporal_alert_rate: float = field(
        default_factory=lambda: _env_float("MAX_TEMPORAL_ALERT_RATE", 0.025)
    )
    max_pr_auc_temporal_drop: float = field(
        default_factory=lambda: _env_float("MAX_PR_AUC_TEMPORAL_DROP", 0.80)
    )
    max_recall_temporal_drop: float = field(
        default_factory=lambda: _env_float("MAX_RECALL_TEMPORAL_DROP", 0.80)
    )
    external_benchmarks_enabled: bool = field(
        default_factory=lambda: _env_bool_alias(
            "RUN_EXTERNAL_BENCHMARKS",
            "EXTERNAL_BENCHMARKS_ENABLED",
            False,
        )
    )
    run_autogluon_benchmark: bool = field(
        default_factory=lambda: _env_bool("RUN_AUTOGLUON_BENCHMARK", True)
    )
    run_h2o_benchmark: bool = field(
        default_factory=lambda: _env_bool("RUN_H2O_BENCHMARK", True)
    )
    run_flaml_benchmark: bool = field(
        default_factory=lambda: _env_bool("RUN_FLAML_BENCHMARK", True)
    )
    external_benchmark_backends: tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "EXTERNAL_BENCHMARK_BACKENDS",
            ("autogluon", "h2o", "flaml"),
        )
    )
    external_benchmark_time_limit_seconds: int = field(
        default_factory=lambda: int(os.getenv("EXTERNAL_BENCHMARK_TIME_LIMIT_SECONDS", "300"))
    )
    external_benchmark_max_models: int = field(
        default_factory=lambda: int(os.getenv("EXTERNAL_BENCHMARK_MAX_MODELS", "10"))
    )
    external_benchmark_fail_fast: bool = field(
        default_factory=lambda: _env_bool("EXTERNAL_BENCHMARK_FAIL_FAST", False)
    )
    random_state: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    out_of_time_size: float = field(
        default_factory=lambda: _env_float("OUT_OF_TIME_SIZE", 0.10)
    )
    threshold_beta: float = 2.0
    min_precision_for_threshold: float = 0.05
    threshold_selection_strategy: str = field(
        default_factory=lambda: os.getenv("THRESHOLD_SELECTION_STRATEGY", "business_cost")
    )
    threshold_analysis_start: float = field(
        default_factory=lambda: _env_float(
            "THRESHOLD_MIN", _env_float("THRESHOLD_ANALYSIS_START", 0.01)
        )
    )
    threshold_analysis_stop: float = field(
        default_factory=lambda: _env_float(
            "THRESHOLD_MAX", _env_float("THRESHOLD_ANALYSIS_STOP", 0.99)
        )
    )
    threshold_analysis_step: float = field(
        default_factory=lambda: _env_float(
            "THRESHOLD_STEP", _env_float("THRESHOLD_ANALYSIS_STEP", 0.01)
        )
    )
    false_positive_cost: float = field(default_factory=lambda: _env_float("FALSE_POSITIVE_COST", 1.0))
    false_negative_cost: float = field(default_factory=lambda: _env_float("FALSE_NEGATIVE_COST", 25.0))
    threshold_cost_scenarios: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: _env_cost_scenarios(
            "THRESHOLD_COST_SCENARIOS",
            ((1.0, 10.0), (1.0, 25.0), (1.0, 50.0), (5.0, 25.0)),
        )
    )
    leakage_roc_auc_warning: float = field(
        default_factory=lambda: _env_float("LEAKAGE_ROC_AUC_WARNING", 0.99)
    )
    strict_leakage_prevention: bool = field(
        default_factory=lambda: _env_bool("STRICT_LEAKAGE_PREVENTION", True)
    )
    promote_baseline: bool = field(default_factory=lambda: _env_bool("PROMOTE_BASELINE", False))
    human_approval_confirmed: bool = field(
        default_factory=lambda: _env_bool("HUMAN_APPROVAL_CONFIRMED", False)
    )
    baseline_overwrite: bool = field(default_factory=lambda: _env_bool("BASELINE_OVERWRITE", False))
    baseline_warning_justification: str | None = field(
        default_factory=lambda: os.getenv("BASELINE_WARNING_JUSTIFICATION")
    )
    baseline_name: str = field(
        default_factory=lambda: os.getenv("BASELINE_NAME", "baseline_after_sampling_fix")
    )
    sampling_fix_applied: bool = field(
        default_factory=lambda: _env_bool("SAMPLING_FIX_APPLIED", True)
    )
    previous_baselines_invalidated_by_sampling_bias: bool = field(
        default_factory=lambda: _env_bool(
            "PREVIOUS_BASELINES_INVALIDATED_BY_SAMPLING_BIAS", True
        )
    )
    promotion_min_recall: float = field(
        default_factory=lambda: _env_float("PROMOTION_MIN_RECALL", 0.90)
    )
    promotion_max_alert_rate: float = field(
        default_factory=lambda: _env_float("PROMOTION_MAX_ALERT_RATE", 0.025)
    )
    promotion_max_oot_pr_auc_drop: float = field(
        default_factory=lambda: _env_float("PROMOTION_MAX_OOT_PR_AUC_DROP", 0.15)
    )
    promotion_max_cost_increase: float = field(
        default_factory=lambda: _env_float("PROMOTION_MAX_COST_INCREASE", 0.05)
    )
    promotion_min_oot_pr_auc_lift: float = field(
        default_factory=lambda: _env_float("PROMOTION_MIN_OOT_PR_AUC_LIFT", 1.0)
    )
    promotion_min_walk_forward_recall: float = field(
        default_factory=lambda: _env_float("PROMOTION_MIN_WALK_FORWARD_RECALL", 0.05)
    )
    promotion_min_walk_forward_pr_auc_lift: float = field(
        default_factory=lambda: _env_float("PROMOTION_MIN_WALK_FORWARD_PR_AUC_LIFT", 1.0)
    )
    promotion_max_walk_forward_recall_drop: float = field(
        default_factory=lambda: _env_float("PROMOTION_MAX_WALK_FORWARD_RECALL_DROP", 0.50)
    )
    dataset_version_override: str | None = field(default_factory=lambda: os.getenv("DATASET_VERSION"))
    feature_set_version: str = field(
        default_factory=lambda: os.getenv("FEATURE_SET_VERSION", "v1")
    )
    code_version_override: str | None = field(default_factory=lambda: os.getenv("CODE_VERSION"))
    run_geo_ablation: bool = field(default_factory=lambda: _env_bool("RUN_GEO_ABLATION", False))
    walk_forward_enabled: bool = field(default_factory=lambda: _env_bool("WALK_FORWARD_ENABLED", False))
    walk_forward_folds: int = field(default_factory=lambda: int(os.getenv("WALK_FORWARD_FOLDS", "4")))
    exclude_geographic_features: bool = field(
        default_factory=lambda: _env_bool("EXCLUDE_GEOGRAPHIC_FEATURES", False)
    )
    feature_exclusions: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("FEATURE_EXCLUSIONS", ())
    )
    geographic_feature_exclusions: tuple[str, ...] = (
        "zip",
        "latitude",
        "longitude",
        "merchant_city",
        "merchant_state",
    )
    categorical_min_frequency: int = field(
        default_factory=lambda: int(os.getenv("CATEGORICAL_MIN_FREQUENCY", "10"))
    )
    raw_data_max_rows: int = field(
        default_factory=lambda: _env_nonnegative_int("RAW_DATA_MAX_ROWS", 0)
    )
    training_max_rows: int | None = field(
        default_factory=lambda: _env_nonnegative_int("TRAINING_MAX_ROWS", 500_000)
    )
    preserve_all_positives: bool = field(
        default_factory=lambda: _env_bool("PRESERVE_ALL_POSITIVES", True)
    )
    negative_sampling_enabled: bool = field(
        default_factory=lambda: _env_bool("NEGATIVE_SAMPLING_ENABLED", True)
    )
    negative_sampling_strategy: str = field(
        default_factory=lambda: os.getenv("NEGATIVE_SAMPLING_STRATEGY", "temporal_stratified")
    )
    negative_sampling_by: str = field(
        default_factory=lambda: os.getenv("NEGATIVE_SAMPLING_BY", "month")
    )
    training_negative_positive_ratio: int | None = field(
        default_factory=lambda: _env_optional_positive_int(
            "NEGATIVE_TO_POSITIVE_RATIO",
            _env_optional_positive_int("TRAINING_NEGATIVE_POSITIVE_RATIO", 100),
        )
    )
    sampling_enforce_fixed_ratio: bool = field(
        default_factory=lambda: _env_bool("SAMPLING_ENFORCE_FIXED_RATIO", True)
    )
    imbalance_strategy: str = field(
        default_factory=lambda: os.getenv("IMBALANCE_STRATEGY", "negative_sampling")
    )
    target_column: str = "is_fraud"
    pipeline_filename: str = "fraud_pipeline.joblib"
    metadata_filename: str = "model_metadata.joblib"
    metadata_json_filename: str = "metadata.json"
    threshold_analysis_filename: str = "threshold_analysis.csv"
    leakage_report_filename: str = "leakage_audit.json"
    threshold_cost_scenarios_filename: str = "threshold_cost_scenarios.csv"
    feature_importance_filename: str = "feature_importance.csv"
    calibration_report_filename: str = "calibration_report.csv"
    score_deciles_filename: str = "score_deciles.csv"
    calibration_metrics_filename: str = "calibration_metrics.json"
    calibration_curve_filename: str = "calibration_curve.png"
    out_of_time_metrics_filename: str = "out_of_time_metrics.json"
    model_card_filename: str = "model_card.md"
    baseline_decision_filename: str = "baseline_decision.json"
    baseline_reference_filename: str = "baseline_after_sampling_fix.json"
    manifest_filename: str = "manifest.json"
    geo_ablation_filename: str = "geo_ablation_results.csv"
    robustness_report_filename: str = "robustness_report.json"
    robustness_markdown_filename: str = "robustness_report.md"
    geo_ablation_report_filename: str = "geo_ablation_report.json"
    geo_ablation_markdown_filename: str = "geo_ablation_report.md"
    target_audit_filename: str = "target_audit.json"
    target_audit_markdown_filename: str = "target_audit.md"
    target_audit_by_split_filename: str = "target_audit_by_split.csv"
    target_audit_by_period_filename: str = "target_audit_by_period.csv"
    sampling_audit_filename: str = "sampling_audit.json"
    sampling_audit_markdown_filename: str = "sampling_audit.md"
    sampling_by_period_filename: str = "sampling_by_period.csv"
    sampling_by_split_filename: str = "sampling_by_split.csv"
    sampling_positive_coverage_filename: str = "sampling_positive_coverage.csv"
    data_drift_report_filename: str = "data_drift_report.json"
    data_drift_markdown_filename: str = "data_drift_report.md"
    data_drift_numeric_filename: str = "data_drift_numeric.csv"
    data_drift_categorical_filename: str = "data_drift_categorical.csv"
    feature_stability_report_filename: str = "feature_stability_report.json"
    feature_stability_markdown_filename: str = "feature_stability_report.md"
    feature_stability_by_period_filename: str = "feature_stability_by_period.csv"
    feature_stability_psi_threshold: float = field(
        default_factory=lambda: _env_float("FEATURE_STABILITY_PSI_THRESHOLD", 0.25)
    )
    walk_forward_report_filename: str = "walk_forward_report.json"
    walk_forward_markdown_filename: str = "walk_forward_report.md"
    model_review_report_filename: str = "model_review_report.md"
    threshold_recommendations_filename: str = "threshold_recommendations.json"
    error_attribution_report_filename: str = "error_attribution_report.json"
    error_attribution_markdown_filename: str = "error_attribution_report.md"
    error_attribution_by_group_filename: str = "error_attribution_by_group.csv"
    performance_by_year_filename: str = "performance_by_year.csv"
    performance_by_month_filename: str = "performance_by_month.csv"
    performance_by_period_markdown_filename: str = "performance_by_period.md"
    top_k_analysis_filename: str = "top_k_analysis.csv"
    top_k_analysis_markdown_filename: str = "top_k_analysis.md"
    top_k_values: tuple[int, ...] = (500, 1000, 2500, 5000, 10000)
    optuna_trials_filename: str = "optuna_trials.csv"
    optuna_study_filename: str = "optuna_study.json"
    objective_score_breakdown_filename: str = "objective_score_breakdown.csv"
    objective_score_breakdown_markdown_filename: str = "objective_score_breakdown.md"
    baseline_challenger_comparison_filename: str = "baseline_challenger_comparison.json"
    baseline_challenger_comparison_markdown_filename: str = "baseline_challenger_comparison.md"
    external_benchmark_filename: str = "external_benchmark_results.csv"
    external_benchmark_summary_filename: str = "external_benchmark_summary.json"
    baseline_pipeline_filename: str = "official_pipeline.joblib"
    baseline_metadata_filename: str = "official_metadata.json"
    training_history_save_pipeline: bool = field(
        default_factory=lambda: _env_bool("TRAINING_HISTORY_SAVE_PIPELINE", True)
    )
    mlflow_tracking_enabled: bool = field(
        default_factory=lambda: _env_bool("MLFLOW_TRACKING_ENABLED", False)
    )
    mlflow_tracking_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))
    mlflow_experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")
    )
    mlflow_artifact_location: str | None = field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_LOCATION")
    )
    mlflow_log_model: bool = field(default_factory=lambda: _env_bool("MLFLOW_LOG_MODEL", True))
    mlflow_register_model: bool = field(
        default_factory=lambda: _env_bool("MLFLOW_REGISTER_MODEL", False)
    )
    mlflow_registered_model_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "fraud-detection-model")
    )
    database_tracking_enabled: bool = field(
        default_factory=lambda: _env_bool("DATABASE_TRACKING_ENABLED", True)
    )
    database_connect_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("DATABASE_CONNECT_TIMEOUT_SECONDS", "3"))
    )
    database_url_override: str | None = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "mlflow"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "mlflow"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "mlflow"))
    storage_backend: str = field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "local"))
    raw_data_prefix: str = field(default_factory=lambda: os.getenv("RAW_DATA_PREFIX", "data/raw"))
    artifacts_prefix: str = field(default_factory=lambda: os.getenv("ARTIFACTS_PREFIX", "artifacts"))
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "fraud-detection"))
    minio_secure: bool = field(default_factory=lambda: _env_bool("MINIO_SECURE", False))
    keep_local_artifacts: bool = field(
        default_factory=lambda: _env_bool("KEEP_LOCAL_ARTIFACTS", True)
    )
    keep_local_raw_data: bool = field(
        default_factory=lambda: _env_bool("KEEP_LOCAL_RAW_DATA", True)
    )
    kaggle_dataset: str = field(
        default_factory=lambda: os.getenv(
            "KAGGLE_DATASET",
            "computingvictor/transactions-fraud-datasets",
        )
    )
    kaggle_overwrite: bool = field(default_factory=lambda: _env_bool("KAGGLE_OVERWRITE", False))
    kaggle_auto_import: bool = field(default_factory=lambda: _env_bool("KAGGLE_AUTO_IMPORT", False))
    kaggle_expected_files: tuple[str, ...] = (
        "transactions_data.csv",
        "cards_data.csv",
        "users_data.csv",
        "mcc_codes.json",
        "train_fraud_labels.json",
    )

    transaction_file_candidates: tuple[str, ...] = (
        "transactions_data.csv",
        "transactions.csv",
        "transaction_data.csv",
    )
    cards_file_candidates: tuple[str, ...] = ("cards_data.csv", "cards_dat.csv", "cards.csv")
    users_file_candidates: tuple[str, ...] = ("users_data.csv", "users.csv", "user_data.csv")
    mcc_file_candidates: tuple[str, ...] = ("mcc_codes.json", "mcc.json")
    labels_file_candidates: tuple[str, ...] = (
        "train_fraud_labels.json",
        "fraud_labels.json",
        "labels.json",
    )
    time_column_candidates: tuple[str, ...] = (
        "date",
        "transaction_date",
        "trans_date_trans_time",
        "timestamp",
        "datetime",
        "time",
    )
    transaction_id_candidates: tuple[str, ...] = ("id", "transaction_id", "trans_id")
    card_id_candidates: tuple[str, ...] = ("card_id", "card", "card_number_id")
    user_id_candidates: tuple[str, ...] = ("client_id", "user_id", "customer_id", "person_id")
    amount_candidates: tuple[str, ...] = ("amount", "amt", "transaction_amount")

    def __post_init__(self) -> None:
        """Derive path defaults after dataclass initialization."""
        if self.raw_data_dir is None:
            object.__setattr__(self, "raw_data_dir", self.project_root / "data" / "raw")
        if self.artifacts_dir is None:
            object.__setattr__(self, "artifacts_dir", self.project_root / "artifacts")
        object.__setattr__(self, "storage_backend", self.storage_backend.strip().lower())
        object.__setattr__(self, "raw_data_prefix", self.raw_data_prefix.strip("/"))
        object.__setattr__(self, "artifacts_prefix", self.artifacts_prefix.strip("/"))
        object.__setattr__(self, "kaggle_dataset", self.kaggle_dataset.strip().strip("/"))
        object.__setattr__(self, "model_name", self.model_name.strip().lower())
        object.__setattr__(
            self,
            "model_selection_engine",
            self.model_selection_engine.strip().lower(),
        )
        object.__setattr__(
            self,
            "external_benchmark_backends",
            tuple(backend.strip().lower() for backend in self.external_benchmark_backends),
        )
        exclusions = tuple(
            item.strip().lower()
            for item in self.feature_exclusions
            if item and item.strip()
        )
        if self.exclude_geographic_features:
            exclusions = tuple(sorted(set(exclusions) | set(self.geographic_feature_exclusions)))
        object.__setattr__(self, "feature_exclusions", exclusions)
        object.__setattr__(
            self,
            "optuna_selection_objective",
            self.optuna_selection_objective.strip().lower(),
        )
        object.__setattr__(
            self,
            "negative_sampling_strategy",
            self.negative_sampling_strategy.strip().lower(),
        )
        object.__setattr__(
            self,
            "negative_sampling_by",
            self.negative_sampling_by.strip().lower(),
        )
        object.__setattr__(
            self,
            "imbalance_strategy",
            self.imbalance_strategy.strip().lower(),
        )
        object.__setattr__(self, "baseline_name", self.baseline_name.strip())
        if self.mlflow_tracking_uri is None or not self.mlflow_tracking_uri.strip():
            object.__setattr__(
                self,
                "mlflow_tracking_uri",
                (self.project_root / "mlruns").resolve().as_uri(),
            )
        else:
            object.__setattr__(self, "mlflow_tracking_uri", self.mlflow_tracking_uri.strip())
        object.__setattr__(
            self,
            "mlflow_experiment_name",
            self.mlflow_experiment_name.strip(),
        )
        if self.mlflow_artifact_location is not None:
            object.__setattr__(
                self,
                "mlflow_artifact_location",
                self.mlflow_artifact_location.strip() or None,
            )
        object.__setattr__(
            self,
            "mlflow_registered_model_name",
            self.mlflow_registered_model_name.strip(),
        )
        if self.model_selection_engine not in {"fixed", "optuna"}:
            raise ValueError("MODEL_SELECTION_ENGINE deve ser 'fixed' ou 'optuna'.")
        if self.storage_backend not in {"local", "minio"}:
            raise ValueError("STORAGE_BACKEND deve ser 'local' ou 'minio'.")
        if self.storage_backend == "local" and (
            not self.keep_local_artifacts or not self.keep_local_raw_data
        ):
            raise ValueError(
                "KEEP_LOCAL_ARTIFACTS e KEEP_LOCAL_RAW_DATA devem ser true "
                "quando STORAGE_BACKEND=local."
            )
        supported_models = set(self.model_params)
        if self.model_name not in supported_models:
            raise ValueError(f"MODEL_NAME nao suportado: {self.model_name}.")
        if not self.optuna_model_candidates:
            raise ValueError("OPTUNA_MODEL_CANDIDATES nao pode ser vazio.")
        unknown_models = set(self.optuna_model_candidates) - supported_models
        if unknown_models:
            raise ValueError(
                "Modelos Optuna nao suportados: " + ", ".join(sorted(unknown_models))
            )
        if self.optuna_trials < 1:
            raise ValueError("OPTUNA_TRIALS deve ser positivo.")
        if self.optuna_trials_per_model < 1:
            raise ValueError("OPTUNA_TRIALS_PER_MODEL deve ser positivo.")
        if self.min_valid_temporal_folds < 1:
            raise ValueError("MIN_VALID_TEMPORAL_FOLDS deve ser positivo.")
        if self.optuna_n_jobs == 0:
            raise ValueError("OPTUNA_N_JOBS nao pode ser zero.")
        if self.optuna_selection_objective not in {
            "validation_pr_auc",
            "temporal_stability",
            "temporal_robustness",
        }:
            raise ValueError(
                "OPTUNA_SELECTION_OBJECTIVE deve ser 'validation_pr_auc', "
                "'temporal_stability' ou 'temporal_robustness'."
            )
        if not self.baseline_name:
            raise ValueError("BASELINE_NAME nao pode ser vazio.")
        if not 0 < self.optuna_temporal_holdout_fraction < 0.5:
            raise ValueError("OPTUNA_TEMPORAL_HOLDOUT_FRACTION deve estar entre 0 e 0.5.")
        if self.optuna_pr_auc_stability_penalty < 0:
            raise ValueError("OPTUNA_PR_AUC_STABILITY_PENALTY nao pode ser negativo.")
        if self.optuna_recall_stability_penalty < 0:
            raise ValueError("OPTUNA_RECALL_STABILITY_PENALTY nao pode ser negativo.")
        if self.optuna_last_window_penalty < 0:
            raise ValueError("OPTUNA_LAST_WINDOW_PENALTY nao pode ser negativo.")
        for name, value in (
            ("MIN_FOLD_RECALL_CANDIDATE", self.min_fold_recall_candidate),
            ("MIN_LAST_FOLD_RECALL_CANDIDATE", self.min_last_fold_recall_candidate),
            ("MAX_TEMPORAL_ALERT_RATE", self.max_temporal_alert_rate),
            ("MAX_PR_AUC_TEMPORAL_DROP", self.max_pr_auc_temporal_drop),
            ("MAX_RECALL_TEMPORAL_DROP", self.max_recall_temporal_drop),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} deve estar entre 0 e 1.")
        if self.min_pr_auc_lift_over_random < 0:
            raise ValueError("MIN_PR_AUC_LIFT_OVER_RANDOM nao pode ser negativo.")
        if self.external_benchmark_time_limit_seconds < 1:
            raise ValueError("EXTERNAL_BENCHMARK_TIME_LIMIT_SECONDS deve ser positivo.")
        if self.external_benchmark_max_models < 1:
            raise ValueError("EXTERNAL_BENCHMARK_MAX_MODELS deve ser positivo.")
        if self.walk_forward_folds < 2:
            raise ValueError("WALK_FORWARD_FOLDS deve ser pelo menos 2.")
        if self.mlflow_tracking_enabled and not self.mlflow_experiment_name:
            raise ValueError("MLFLOW_EXPERIMENT_NAME nao pode ser vazio.")
        if self.mlflow_register_model and not self.mlflow_log_model:
            raise ValueError("MLFLOW_REGISTER_MODEL exige MLFLOW_LOG_MODEL=true.")
        if self.mlflow_register_model and not self.mlflow_registered_model_name:
            raise ValueError("MLFLOW_REGISTERED_MODEL_NAME nao pode ser vazio.")
        supported_benchmarks = {"autogluon", "h2o", "flaml"}
        unknown_benchmarks = set(self.external_benchmark_backends) - supported_benchmarks
        if unknown_benchmarks:
            raise ValueError(
                "Benchmarks externos nao suportados: "
                + ", ".join(sorted(unknown_benchmarks))
            )
        strategy = self.threshold_selection_strategy.strip().lower()
        if strategy not in {"business_cost", "fbeta"}:
            raise ValueError("THRESHOLD_SELECTION_STRATEGY deve ser 'business_cost' ou 'fbeta'.")
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("Custos de falso positivo e falso negativo nao podem ser negativos.")
        if any(fp < 0 or fn < 0 for fp, fn in self.threshold_cost_scenarios):
            raise ValueError("Cenarios de custo nao podem conter valores negativos.")
        if not 0 <= self.threshold_analysis_start < self.threshold_analysis_stop <= 1:
            raise ValueError(
                "THRESHOLD_ANALYSIS_START e STOP devem respeitar 0 <= START < STOP <= 1."
            )
        if self.threshold_analysis_step <= 0:
            raise ValueError("THRESHOLD_ANALYSIS_STEP deve ser positivo.")
        if not 0 <= self.leakage_roc_auc_warning <= 1:
            raise ValueError("LEAKAGE_ROC_AUC_WARNING deve estar entre 0 e 1.")
        split_total = self.validation_size + self.test_size + self.out_of_time_size
        if not 0 < split_total < 1:
            raise ValueError("A soma dos splits de validacao, teste e OOT deve estar entre 0 e 1.")
        if not 0 <= self.promotion_min_recall <= 1:
            raise ValueError("PROMOTION_MIN_RECALL deve estar entre 0 e 1.")
        if not 0 <= self.promotion_max_alert_rate <= 1:
            raise ValueError("PROMOTION_MAX_ALERT_RATE deve estar entre 0 e 1.")
        if not 0 <= self.promotion_max_oot_pr_auc_drop <= 1:
            raise ValueError("PROMOTION_MAX_OOT_PR_AUC_DROP deve estar entre 0 e 1.")
        if self.promotion_max_cost_increase < 0:
            raise ValueError("PROMOTION_MAX_COST_INCREASE nao pode ser negativo.")
        if self.promotion_min_oot_pr_auc_lift < 0:
            raise ValueError("PROMOTION_MIN_OOT_PR_AUC_LIFT nao pode ser negativo.")
        if not 0 <= self.promotion_min_walk_forward_recall <= 1:
            raise ValueError("PROMOTION_MIN_WALK_FORWARD_RECALL deve estar entre 0 e 1.")
        if self.promotion_min_walk_forward_pr_auc_lift < 0:
            raise ValueError("PROMOTION_MIN_WALK_FORWARD_PR_AUC_LIFT nao pode ser negativo.")
        if not 0 <= self.promotion_max_walk_forward_recall_drop <= 1:
            raise ValueError("PROMOTION_MAX_WALK_FORWARD_RECALL_DROP deve estar entre 0 e 1.")
        if self.feature_stability_psi_threshold < 0:
            raise ValueError("FEATURE_STABILITY_PSI_THRESHOLD nao pode ser negativo.")
        if self.raw_data_max_rows < 0:
            raise ValueError("RAW_DATA_MAX_ROWS deve ser zero ou positivo.")
        if self.training_max_rows is not None and self.training_max_rows < 0:
            raise ValueError("TRAINING_MAX_ROWS deve ser zero, positivo ou None.")
        if not self.preserve_all_positives:
            raise ValueError("PRESERVE_ALL_POSITIVES deve permanecer true para treino governado.")
        if self.negative_sampling_strategy != "temporal_stratified":
            raise ValueError(
                "NEGATIVE_SAMPLING_STRATEGY deve ser 'temporal_stratified'."
            )
        if self.negative_sampling_by not in {"month", "year"}:
            raise ValueError("NEGATIVE_SAMPLING_BY deve ser 'month' ou 'year'.")
        if (
            self.training_negative_positive_ratio is not None
            and self.training_negative_positive_ratio < 1
        ):
            raise ValueError("NEGATIVE_TO_POSITIVE_RATIO deve ser pelo menos 1.")
        if not self.negative_sampling_enabled and self.training_max_rows not in {None, 0}:
            raise ValueError(
                "TRAINING_MAX_ROWS exige NEGATIVE_SAMPLING_ENABLED=true; "
                "limite sequencial nao e permitido."
            )
        supported_imbalance_strategies = {
            "negative_sampling",
            "class_weight",
            "sample_weight",
            "negative_sampling_plus_class_weight",
        }
        if self.imbalance_strategy not in supported_imbalance_strategies:
            raise ValueError(
                "IMBALANCE_STRATEGY invalida. Opcoes: "
                + ", ".join(sorted(supported_imbalance_strategies))
            )
        if (
            self.imbalance_strategy in {"negative_sampling", "negative_sampling_plus_class_weight"}
            and not self.negative_sampling_enabled
        ):
            raise ValueError("IMBALANCE_STRATEGY selecionada exige NEGATIVE_SAMPLING_ENABLED=true.")
        object.__setattr__(self, "threshold_selection_strategy", strategy)

    @property
    def external_benchmark_backend_flags(self) -> dict[str, bool]:
        """Return per-backend switches for optional external benchmarks."""
        return {
            "autogluon": self.run_autogluon_benchmark,
            "h2o": self.run_h2o_benchmark,
            "flaml": self.run_flaml_benchmark,
        }

    @property
    def enabled_external_benchmark_backends(self) -> tuple[str, ...]:
        """Backends selected by the legacy list and the per-backend run flags."""
        flags = self.external_benchmark_backend_flags
        return tuple(
            backend
            for backend in self.external_benchmark_backends
            if flags.get(backend, False)
        )

    @property
    def pipeline_path(self) -> Path:
        """Path where the fitted sklearn pipeline is stored."""
        return self.artifacts_dir / self.pipeline_filename

    @property
    def metadata_path(self) -> Path:
        """Path where threshold and metrics metadata are stored."""
        return self.artifacts_dir / self.metadata_filename

    @property
    def threshold_analysis_path(self) -> Path:
        """Path where threshold comparison rows are stored."""
        return self.artifacts_dir / self.threshold_analysis_filename

    @property
    def leakage_report_path(self) -> Path:
        """Path where the leakage audit is stored."""
        return self.artifacts_dir / self.leakage_report_filename

    def artifact_path(self, filename: str) -> Path:
        """Return a path inside the current artifacts directory."""
        return self.artifacts_dir / filename

    @property
    def governance_artifact_filenames(self) -> tuple[str, ...]:
        """Artifacts expected for a governed completed training run."""
        return (
            self.pipeline_filename,
            self.metadata_filename,
            self.metadata_json_filename,
            self.threshold_analysis_filename,
            self.threshold_cost_scenarios_filename,
            self.leakage_report_filename,
            self.feature_importance_filename,
            self.calibration_report_filename,
            self.score_deciles_filename,
            self.calibration_metrics_filename,
            self.calibration_curve_filename,
            self.out_of_time_metrics_filename,
            self.model_card_filename,
            self.baseline_decision_filename,
            self.baseline_reference_filename,
            self.manifest_filename,
            self.geo_ablation_filename,
            self.robustness_report_filename,
            self.robustness_markdown_filename,
            self.geo_ablation_report_filename,
            self.geo_ablation_markdown_filename,
            self.target_audit_filename,
            self.target_audit_markdown_filename,
            self.target_audit_by_split_filename,
            self.target_audit_by_period_filename,
            self.sampling_audit_filename,
            self.sampling_audit_markdown_filename,
            self.sampling_by_period_filename,
            self.sampling_by_split_filename,
            self.sampling_positive_coverage_filename,
            self.data_drift_report_filename,
            self.data_drift_markdown_filename,
            self.data_drift_numeric_filename,
            self.data_drift_categorical_filename,
            self.feature_stability_report_filename,
            self.feature_stability_markdown_filename,
            self.feature_stability_by_period_filename,
            self.walk_forward_report_filename,
            self.walk_forward_markdown_filename,
            self.model_review_report_filename,
            self.threshold_recommendations_filename,
            self.error_attribution_report_filename,
            self.error_attribution_markdown_filename,
            self.error_attribution_by_group_filename,
            self.performance_by_year_filename,
            self.performance_by_month_filename,
            self.performance_by_period_markdown_filename,
            self.top_k_analysis_filename,
            self.top_k_analysis_markdown_filename,
            self.optuna_trials_filename,
            self.optuna_study_filename,
            self.objective_score_breakdown_filename,
            self.objective_score_breakdown_markdown_filename,
            self.baseline_challenger_comparison_filename,
            self.baseline_challenger_comparison_markdown_filename,
            self.external_benchmark_filename,
            self.external_benchmark_summary_filename,
        )

    @property
    def baseline_dir(self) -> Path:
        """Directory containing the promoted official baseline."""
        return self.artifacts_dir / "baseline"

    @property
    def training_history_dir(self) -> Path:
        """Directory containing immutable training executions."""
        return self.artifacts_dir / "history"

    @property
    def training_history_index_path(self) -> Path:
        """CSV index used to compare historical training runs."""
        return self.training_history_dir / "runs.csv"

    @property
    def database_url(self) -> str:
        """PostgreSQL URL used by tracking repositories and migrations."""
        if self.database_url_override:
            return self.database_url_override
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        database = quote_plus(self.postgres_database)
        return (
            f"postgresql+psycopg://{user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{database}"
        )

    def raw_object_key(self, filename: str) -> str:
        """Return the object key for a raw dataset file."""
        return f"{self.raw_data_prefix}/{filename}".strip("/")

    def artifact_object_key(self, filename: str) -> str:
        """Return the object key for a model artifact."""
        return f"{self.artifacts_prefix}/{filename}".strip("/")

    def object_uri(self, key: str) -> str:
        """Return the persistent URI for an object key."""
        if self.storage_backend == "minio":
            return f"minio://{self.minio_bucket}/{key.lstrip('/')}"
        return (self.project_root / key).resolve().as_uri()

    @property
    def pipeline_object_key(self) -> str:
        """Object key for the fitted sklearn pipeline."""
        return self.artifact_object_key(self.pipeline_filename)

    @property
    def metadata_object_key(self) -> str:
        """Object key for threshold and metrics metadata."""
        return self.artifact_object_key(self.metadata_filename)

    @property
    def model_params(self) -> dict[str, dict[str, Any]]:
        """Default model hyperparameters used by ModelFactory."""
        params = {
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": self.random_state,
            },
            "logistic_regression_regularized": {
                "C": 0.1,
                "max_iter": 1500,
                "class_weight": "balanced",
                "solver": "liblinear",
                "random_state": self.random_state,
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": self.random_state,
            },
            "random_forest_regularized": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": self.random_state,
            },
            "extra_trees_regularized": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "n_jobs": -1,
                "random_state": self.random_state,
            },
            "balanced_random_forest": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "n_jobs": -1,
                "random_state": self.random_state,
            },
            "easy_ensemble": {
                "n_estimators": 20,
                "n_jobs": -1,
                "random_state": self.random_state,
            },
            "rus_boost": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "random_state": self.random_state,
            },
            "hist_gradient_boosting": {
                "learning_rate": 0.05,
                "max_iter": 250,
                "l2_regularization": 0.1,
                "class_weight": "balanced",
                "random_state": self.random_state,
            },
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": self.random_state,
                "verbosity": 0,
            },
            "xgboost_scale_pos_weight": {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "hist",
                "n_jobs": -1,
                "random_state": self.random_state,
                "verbosity": 0,
            },
            "lightgbm": {
                "objective": "binary",
                "class_weight": "balanced",
                "n_jobs": -1,
                "random_state": self.random_state,
                "verbosity": -1,
            },
            "lightgbm_scale_pos_weight": {
                "objective": "binary",
                "n_jobs": -1,
                "random_state": self.random_state,
                "verbosity": -1,
            },
            "catboost": {
                "loss_function": "Logloss",
                "eval_metric": "PRAUC",
                "auto_class_weights": "Balanced",
                "allow_writing_files": False,
                "verbose": False,
                "random_seed": self.random_state,
                "thread_count": -1,
            },
            "catboost_regularized": {
                "loss_function": "Logloss",
                "eval_metric": "PRAUC",
                "auto_class_weights": "Balanced",
                "l2_leaf_reg": 10.0,
                "allow_writing_files": False,
                "verbose": False,
                "random_seed": self.random_state,
                "thread_count": -1,
            },
        }
        if self.imbalance_strategy in {"negative_sampling", "sample_weight"}:
            for model_params in params.values():
                model_params.pop("class_weight", None)
                model_params.pop("auto_class_weights", None)
            for name in ("xgboost_scale_pos_weight", "lightgbm_scale_pos_weight"):
                params[name].pop("scale_pos_weight", None)
        return params
