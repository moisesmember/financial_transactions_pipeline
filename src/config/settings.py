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


def _env_float(name: str, default: float) -> float:
    """Parse a floating-point environment variable."""
    value = os.getenv(name)
    return default if value is None else float(value.strip())


@dataclass(frozen=True)
class Settings:
    """Centralized configuration for paths, data columns and modeling."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    raw_data_dir: Path | None = None
    artifacts_dir: Path | None = None
    model_name: str = "logistic_regression"
    random_state: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    threshold_beta: float = 2.0
    min_precision_for_threshold: float = 0.05
    threshold_selection_strategy: str = field(
        default_factory=lambda: os.getenv("THRESHOLD_SELECTION_STRATEGY", "business_cost")
    )
    threshold_analysis_start: float = field(
        default_factory=lambda: _env_float("THRESHOLD_ANALYSIS_START", 0.08)
    )
    threshold_analysis_stop: float = field(
        default_factory=lambda: _env_float("THRESHOLD_ANALYSIS_STOP", 0.30)
    )
    threshold_analysis_step: float = field(
        default_factory=lambda: _env_float("THRESHOLD_ANALYSIS_STEP", 0.01)
    )
    false_positive_cost: float = field(default_factory=lambda: _env_float("FALSE_POSITIVE_COST", 1.0))
    false_negative_cost: float = field(default_factory=lambda: _env_float("FALSE_NEGATIVE_COST", 25.0))
    leakage_roc_auc_warning: float = field(
        default_factory=lambda: _env_float("LEAKAGE_ROC_AUC_WARNING", 0.99)
    )
    strict_leakage_prevention: bool = field(
        default_factory=lambda: _env_bool("STRICT_LEAKAGE_PREVENTION", True)
    )
    promote_baseline: bool = field(default_factory=lambda: _env_bool("PROMOTE_BASELINE", False))
    baseline_overwrite: bool = field(default_factory=lambda: _env_bool("BASELINE_OVERWRITE", False))
    training_max_rows: int | None = field(
        default_factory=lambda: _env_optional_positive_int("TRAINING_MAX_ROWS", 500_000)
    )
    target_column: str = "is_fraud"
    pipeline_filename: str = "fraud_pipeline.joblib"
    metadata_filename: str = "model_metadata.joblib"
    threshold_analysis_filename: str = "threshold_analysis.csv"
    leakage_report_filename: str = "leakage_audit.json"
    baseline_pipeline_filename: str = "official_pipeline.joblib"
    baseline_metadata_filename: str = "official_metadata.json"
    training_history_save_pipeline: bool = field(
        default_factory=lambda: _env_bool("TRAINING_HISTORY_SAVE_PIPELINE", True)
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
        strategy = self.threshold_selection_strategy.strip().lower()
        if strategy not in {"business_cost", "fbeta"}:
            raise ValueError("THRESHOLD_SELECTION_STRATEGY deve ser 'business_cost' ou 'fbeta'.")
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("Custos de falso positivo e falso negativo nao podem ser negativos.")
        object.__setattr__(self, "threshold_selection_strategy", strategy)

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
        return {
            "logistic_regression": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "solver": "lbfgs",
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
            "hist_gradient_boosting": {
                "learning_rate": 0.05,
                "max_iter": 250,
                "l2_regularization": 0.1,
                "random_state": self.random_state,
            },
        }
