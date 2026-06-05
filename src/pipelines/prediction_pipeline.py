"""Batch and service prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config.settings import Settings
from src.storage.sync import StorageSyncService


class FraudPredictionService:
    """Load the persisted pipeline and generate fraud predictions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensure_local_artifacts()
        self.pipeline = joblib.load(self.settings.pipeline_path)
        self.metadata = joblib.load(self.settings.metadata_path) if self.settings.metadata_path.exists() else {}
        self.threshold = float(self.metadata.get("threshold", 0.5))

    def _ensure_local_artifacts(self) -> None:
        """Ensure model artifacts are available locally, downloading from MinIO if needed."""
        if not self.settings.pipeline_path.exists():
            StorageSyncService(self.settings).download_artifact(
                self.settings.pipeline_object_key, self.settings.pipeline_path
            )
        if not self.settings.metadata_path.exists():
            StorageSyncService(self.settings).download_artifact(
                self.settings.metadata_object_key, self.settings.metadata_path
            )
        if not self.settings.pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline nao encontrada em {self.settings.pipeline_path}. Execute python main.py primeiro."
            )

    def predict_frame(self, records: pd.DataFrame) -> pd.DataFrame:
        """Predict fraud scores and labels for a dataframe."""
        scores = self.pipeline.predict_proba(records)[:, 1]
        output = records.copy()
        output["fraud_score"] = scores
        output["is_fraud_predicted"] = (scores >= self.threshold).astype(int)
        output["threshold"] = self.threshold
        return output

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict fraud for API-style records."""
        frame = pd.DataFrame(records)
        predictions = self.predict_frame(frame)
        return predictions[["fraud_score", "is_fraud_predicted", "threshold"]].to_dict(orient="records")

    def predict_csv(self, input_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
        """Read a CSV, score it and optionally persist predictions."""
        frame = pd.read_csv(input_path)
        predictions = self.predict_frame(frame)
        if output_path is not None:
            predictions.to_csv(output_path, index=False)
        return predictions
