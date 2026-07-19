"""Register an existing rejected run as the post-sampling-fix reference baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import Settings
from src.models.governance_artifacts import write_manifest
from src.storage.factory import create_object_store
from src.storage.postgres_training_history import PostgresTrainingHistoryRepository


def register(mlflow_run_id: str, training_run_id: str, baseline_name: str) -> None:
    """Update MLflow, MinIO and PostgreSQL without promoting the rejected model."""
    settings = Settings()
    store = create_object_store(settings)
    history_prefix = settings.artifact_object_key(f"history/{training_run_id}")
    with TemporaryDirectory(prefix="sampling-fix-baseline-") as temporary:
        artifact_root = Path(temporary) / "artifacts"
        run_dir = artifact_root / "history" / training_run_id
        run_dir.mkdir(parents=True)
        objects = list(
            store.client.list_objects(
                store.bucket,
                prefix=history_prefix,
                recursive=True,
            )
        )
        if not objects:
            raise FileNotFoundError(f"Historico MinIO nao encontrado: {history_prefix}")
        for item in objects:
            relative = Path(item.object_name).relative_to(history_prefix)
            store.download_file(item.object_name, run_dir / relative)

        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        decision = (metadata.get("baseline_decision") or {}).get("decision")
        if decision != "reject":
            raise ValueError(
                "Este registrador preserva somente baseline analitico rejeitado; "
                f"decisao encontrada={decision}."
            )
        metadata.update(
            {
                "baseline_name": baseline_name,
                "sampling_fix_applied": True,
                "previous_baselines_invalidated_by_sampling_bias": True,
                "human_approval_confirmed": False,
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True, allow_nan=False),
            encoding="utf-8",
        )
        reference = {
            "baseline_name": baseline_name,
            "sampling_fix_applied": True,
            "previous_baselines_invalidated_by_sampling_bias": True,
            "training_run_id": training_run_id,
            "mlflow_run_id": mlflow_run_id,
            "model_name": metadata.get("model_name"),
            "threshold": metadata.get("threshold"),
            "decision": decision,
            "notes": [
                "As rodadas anteriores tinham vies temporal causado por limite sequencial.",
                "Esta rodada e o novo baseline de referencia apos correcao da amostragem.",
                "O modelo continua rejeitado e nao foi promovido para producao.",
            ],
        }
        reference_path = run_dir / settings.baseline_reference_filename
        reference_path.write_text(
            json.dumps(reference, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        review_path = run_dir / settings.model_review_report_filename
        review = review_path.read_text(encoding="utf-8")
        if "## What changed after sampling fix" not in review:
            review += (
                "\n## What changed after sampling fix\n\n"
                "- A amostragem sequencial anterior foi corrigida.\n"
                "- As rodadas antigas nao devem ser usadas como baseline definitivo.\n"
                f"- `{baseline_name}` e o novo baseline analitico, ainda com decisao `reject`.\n"
                "- A proxima otimizacao deve focar robustez temporal, falsos negativos OOT e features instaveis.\n"
            )
            review_path.write_text(review, encoding="utf-8")

        write_manifest(run_dir / settings.manifest_filename, list(run_dir.iterdir()))
        for path in (
            metadata_path,
            reference_path,
            review_path,
            run_dir / settings.manifest_filename,
        ):
            store.upload_file(path, f"{history_prefix}/{path.name}")
        store.upload_file(
            reference_path,
            settings.artifact_object_key(f"baselines/{baseline_name}.json"),
        )

        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()
        values = {
            "baseline_name": baseline_name,
            "sampling_fix_applied": "true",
            "previous_baselines_invalidated_by_sampling_bias": "true",
        }
        for key, value in values.items():
            client.log_param(mlflow_run_id, key, value)
            client.set_tag(mlflow_run_id, key, value)
        client.log_artifact(
            mlflow_run_id,
            str(reference_path),
            artifact_path="governance",
        )

        persistence_settings = replace(settings, artifacts_dir=artifact_root)
        persisted = PostgresTrainingHistoryRepository(
            persistence_settings
        ).persist_if_available(run_dir)
        if not persisted:
            raise RuntimeError("Falha ao persistir o baseline no PostgreSQL.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--baseline-name", default="baseline_after_sampling_fix")
    args = parser.parse_args()
    register(args.mlflow_run_id, args.training_run_id, args.baseline_name)


if __name__ == "__main__":
    main()
