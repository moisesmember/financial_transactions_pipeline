"""DTO schemas for the fraud prediction API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    promote_baseline: bool | None = None
    baseline_overwrite: bool | None = None
    run_geo_ablation: bool | None = None
    training_history_save_pipeline: bool | None = None
    training_max_rows: int | None = Field(default=None, ge=0)
    baseline_warning_justification: str | None = None
    promotion_min_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_alert_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_oot_pr_auc_drop: float | None = Field(default=None, ge=0.0, le=1.0)
    promotion_max_cost_increase: float | None = Field(default=None, ge=0.0)

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
        if overrides.get("training_max_rows") == 0:
            overrides["training_max_rows"] = None
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
