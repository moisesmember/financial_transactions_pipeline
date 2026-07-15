"""DTO schemas for the fraud prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    """Batch prediction request with flexible transaction records."""

    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    """Single transaction prediction result."""

    fraud_score: float
    is_fraud_predicted: int
    threshold: float


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


class ModelRunExportResponse(BaseModel):
    """Paginated export from the model run fact view."""

    source: str
    total: int
    count: int
    limit: int
    offset: int
    items: list[dict[str, Any]]


class TrainingRequest(BaseModel):
    """Optional training overrides; omitted fields keep their Settings/.env values."""

    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.upper(),
        populate_by_name=True,
        extra="forbid",
    )

    threshold_selection_strategy: Literal["business_cost", "fbeta"] | None = None
    threshold_analysis_start: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_analysis_stop: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_analysis_step: float | None = Field(default=None, gt=0.0, le=1.0)
    false_positive_cost: float | None = Field(default=None, ge=0.0)
    false_negative_cost: float | None = Field(default=None, ge=0.0)
    threshold_cost_scenarios: tuple[tuple[float, float], ...] | None = None
    out_of_time_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    leakage_roc_auc_warning: float | None = Field(default=None, ge=0.0, le=1.0)
    strict_leakage_prevention: bool | None = None
    external_benchmarks_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("EXTERNAL_BENCHMARKS_ENABLED", "RUN_EXTERNAL_BENCHMARKS"),
    )
    run_autogluon_benchmark: bool | None = None
    run_h2o_benchmark: bool | None = None
    run_flaml_benchmark: bool | None = None
    promote_baseline: bool | None = None
    baseline_overwrite: bool | None = None
    run_geo_ablation: bool | None = None
    walk_forward_enabled: bool | None = None
    walk_forward_folds: int | None = Field(default=None, ge=2)
    exclude_geographic_features: bool | None = None
    feature_exclusions: tuple[str, ...] | None = None
    optuna_selection_objective: Literal["validation_pr_auc", "temporal_stability"] | None = None
    optuna_temporal_holdout_fraction: float | None = Field(default=None, gt=0.0, lt=0.5)
    optuna_pr_auc_stability_penalty: float | None = Field(default=None, ge=0.0)
    optuna_recall_stability_penalty: float | None = Field(default=None, ge=0.0)
    optuna_last_window_penalty: float | None = Field(default=None, ge=0.0)
    training_history_save_pipeline: bool | None = None
    mlflow_tracking_enabled: bool | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_artifact_location: str | None = None
    mlflow_log_model: bool | None = None
    mlflow_register_model: bool | None = None
    mlflow_registered_model_name: str | None = None
    raw_data_max_rows: int | None = Field(default=None, ge=0)
    training_max_rows: int | None = Field(default=None, ge=0)
    preserve_all_positives: bool | None = None
    negative_sampling_enabled: bool | None = None
    negative_sampling_strategy: Literal["temporal_stratified"] | None = None
    negative_sampling_by: Literal["month", "year"] | None = None
    training_negative_positive_ratio: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "TRAINING_NEGATIVE_POSITIVE_RATIO",
            "NEGATIVE_TO_POSITIVE_RATIO",
        ),
    )
    baseline_warning_justification: str | None = None
    promotion_min_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_alert_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_oot_pr_auc_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_cost_increase: float | None = Field(default=None, ge=0.0)
    promotion_min_oot_pr_auc_lift: float | None = Field(default=None, ge=0.0)
    promotion_min_walk_forward_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_min_walk_forward_pr_auc_lift: float | None = Field(default=None, ge=0.0)
    promotion_max_walk_forward_recall_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_stability_psi_threshold: float | None = Field(default=None, ge=0.0)

    @field_validator("threshold_cost_scenarios", mode="before")
    @classmethod
    def parse_threshold_cost_scenarios(cls, value: Any) -> Any:
        """Accept the .env notation or a JSON array of FP/FN pairs."""
        if value is None or not isinstance(value, str):
            return value
        scenarios: list[tuple[float, float]] = []
        for item in value.split(","):
            try:
                false_positive_cost, false_negative_cost = item.strip().split(":", maxsplit=1)
                scenarios.append((float(false_positive_cost), float(false_negative_cost)))
            except ValueError as exc:
                raise ValueError(
                    "THRESHOLD_COST_SCENARIOS deve usar o formato `1:10,1:25`."
                ) from exc
        return tuple(scenarios)

    @field_validator("feature_exclusions", mode="before")
    @classmethod
    def parse_feature_exclusions(cls, value: Any) -> Any:
        """Accept .env-style comma-separated feature exclusions."""
        if value is None or not isinstance(value, str):
            return value
        return tuple(item.strip().lower() for item in value.split(",") if item.strip())

    @model_validator(mode="after")
    def validate_threshold_range(self) -> "TrainingRequest":
        """Reject an explicitly inverted threshold range."""
        if (
            self.threshold_analysis_start is not None
            and self.threshold_analysis_stop is not None
            and self.threshold_analysis_start >= self.threshold_analysis_stop
        ):
            raise ValueError("THRESHOLD_ANALYSIS_START deve ser menor que STOP.")
        return self

    def settings_overrides(self) -> dict[str, Any]:
        """Return explicit settings overrides, preserving .env defaults for omissions."""
        overrides = self.model_dump(exclude_none=True)
        if overrides.get("training_negative_positive_ratio") == 0:
            overrides["training_negative_positive_ratio"] = None
        return overrides


class TrainingJobResponse(BaseModel):
    """Current state and eventual result of an asynchronous training job."""

    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    configuration: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class TrainingReportResponse(BaseModel):
    """Transparent governed report for one completed model training run."""

    run_id: str
    generated_at: datetime
    executive_summary: list[str]
    run: dict[str, Any]
    model: dict[str, Any]
    dataset: dict[str, Any]
    performance: dict[str, Any]
    features: dict[str, Any]
    audit: dict[str, Any]
    model_search: dict[str, Any]
    external_benchmarks: list[dict[str, Any]]
    robustness_experiments: list[dict[str, Any]]
    threshold_analysis: dict[str, Any]
    artifacts: dict[str, Any]
    baseline: dict[str, Any]
