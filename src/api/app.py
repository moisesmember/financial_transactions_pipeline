"""FastAPI application for fraud scoring."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status

from src.api.schemas import (
    HealthResponse,
    ModelRunExportResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.config.settings import Settings
from src.pipelines.prediction_pipeline import FraudPredictionService
from src.storage.model_run_fact_repository import (
    QUALIFIED_VIEW,
    ModelRunFactRepository,
    ModelRunFactRepositoryError,
    ModelRunFactViewNotFoundError,
)


app = FastAPI(title="Financial Fraud Detection API", version="1.0.0")


@lru_cache(maxsize=1)
def get_prediction_service() -> FraudPredictionService:
    """Create and cache the prediction service."""
    return FraudPredictionService(Settings())


@lru_cache(maxsize=1)
def get_model_run_fact_repository() -> ModelRunFactRepository:
    """Create and cache the read-only model tracking repository."""
    return ModelRunFactRepository(Settings())


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


@app.get(
    "/model-runs/export",
    response_model=ModelRunExportResponse,
    summary="Export model run history as JSON",
)
def export_model_runs(
    response: Response,
    repository: Annotated[ModelRunFactRepository, Depends(get_model_run_fact_repository)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ModelRunExportResponse:
    """Export all columns from the consolidated model run fact view."""
    try:
        total, items = repository.export_page(limit=limit, offset=offset)
    except ModelRunFactViewNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"{exc} Execute `python -m scripts.migrate_database upgrade` "
                "antes de usar esta rota."
            ),
        ) from exc
    except ModelRunFactRepositoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PostgreSQL indisponivel para exportacao do historico de modelos.",
        ) from exc

    response.headers["Content-Disposition"] = 'attachment; filename="fact_model_runs.json"'
    return ModelRunExportResponse(
        source=QUALIFIED_VIEW,
        total=total,
        count=len(items),
        limit=limit,
        offset=offset,
        items=items,
    )
