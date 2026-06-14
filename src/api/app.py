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
    TrainingJobResponse,
    TrainingRequest,
)
from src.api.training_service import (
    TrainingAlreadyRunningError,
    TrainingJobManager,
    TrainingJobNotFoundError,
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


@lru_cache(maxsize=1)
def get_training_job_manager() -> TrainingJobManager:
    """Create the process-local manager for resource-intensive training jobs."""
    return TrainingJobManager(on_complete=get_prediction_service.cache_clear)


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


@app.post(
    "/training-runs",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a governed model training run",
)
def start_training(
    manager: Annotated[TrainingJobManager, Depends(get_training_job_manager)],
    request: TrainingRequest | None = None,
) -> TrainingJobResponse:
    """Start asynchronous training with request overrides and .env defaults."""
    try:
        return TrainingJobResponse.model_validate(manager.start(request or TrainingRequest()))
    except TrainingAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ja existe um treinamento em execucao: {exc}.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@app.get(
    "/training-runs/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get model training status",
)
def get_training_status(
    job_id: str,
    manager: Annotated[TrainingJobManager, Depends(get_training_job_manager)],
) -> TrainingJobResponse:
    """Return queue, execution and result details for one API training job."""
    try:
        return TrainingJobResponse.model_validate(manager.get(job_id))
    except TrainingJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Treinamento nao encontrado: {job_id}.",
        ) from exc


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
