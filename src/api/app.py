"""FastAPI application for fraud scoring."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.config.settings import Settings
from src.pipelines.prediction_pipeline import FraudPredictionService


app = FastAPI(title="Financial Fraud Detection API", version="1.0.0")


@lru_cache(maxsize=1)
def get_prediction_service() -> FraudPredictionService:
    """Create and cache the prediction service."""
    return FraudPredictionService(Settings())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health information."""
    try:
        get_prediction_service()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="model_not_found", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one or more transactions."""
    service = get_prediction_service()
    predictions = service.predict_records(request.records)
    return PredictionResponse(predictions=predictions)
