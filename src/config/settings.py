"""Application settings for the fraud detection project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    target_column: str = "is_fraud"
    pipeline_filename: str = "fraud_pipeline.joblib"
    metadata_filename: str = "model_metadata.joblib"

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

    @property
    def pipeline_path(self) -> Path:
        """Path where the fitted sklearn pipeline is stored."""
        return self.artifacts_dir / self.pipeline_filename

    @property
    def metadata_path(self) -> Path:
        """Path where threshold and metrics metadata are stored."""
        return self.artifacts_dir / self.metadata_filename

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
