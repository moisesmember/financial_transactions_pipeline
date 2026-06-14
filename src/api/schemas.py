"""DTO schemas for the fraud prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
