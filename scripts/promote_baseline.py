"""Promote the current trained model to the official baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import joblib

from src.config.settings import Settings
from src.models.baseline import BaselineRegistry
from src.storage.postgres_training_history import PostgresTrainingHistoryRepository
from src.storage.sync import StorageSyncService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Substitui o baseline oficial existente.")
    parser.add_argument("--run-id", help="Promove uma execucao especifica do historico.")
    return parser.parse_args()


def main() -> None:
    """Promote current pipeline and metadata artifacts."""
    args = parse_args()
    settings = replace(Settings(), promote_baseline=True)
    storage_sync = StorageSyncService(settings)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    storage_sync.prepare_artifact_workspace()
    if settings.database_tracking_enabled and not args.run_id:
        raise ValueError(
            "Com DATABASE_TRACKING_ENABLED=true, informe --run-id para manter "
            "baseline local e PostgreSQL sincronizados."
        )
    pipeline_path = settings.pipeline_path
    run_dir = None
    if args.run_id:
        run_dir = settings.training_history_dir / args.run_id
        if settings.storage_backend == "minio":
            filenames = {
                "metadata.json",
                *settings.governance_artifact_filenames,
            }
            for filename in filenames:
                storage_sync.download_artifact(
                    settings.artifact_object_key(f"history/{args.run_id}/{filename}"),
                    run_dir / filename,
                )
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Run historico nao encontrado: {args.run_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pipeline_path = run_dir / settings.pipeline_filename
        if not pipeline_path.exists():
            raise FileNotFoundError(
                "O pipeline nao foi arquivado para esse run. "
                "Use TRAINING_HISTORY_SAVE_PIPELINE=true nos proximos treinos."
            )
        report_paths = [
            run_dir / filename
            for filename in settings.governance_artifact_filenames
            if filename not in {settings.pipeline_filename, settings.metadata_filename}
        ]
    else:
        if settings.storage_backend == "minio":
            for filename in settings.governance_artifact_filenames:
                storage_sync.download_artifact(
                    settings.artifact_object_key(filename),
                    settings.artifact_path(filename),
                )
        metadata = joblib.load(settings.metadata_path)
        report_paths = [
            settings.artifact_path(filename)
            for filename in settings.governance_artifact_filenames
            if filename not in {settings.pipeline_filename, settings.metadata_filename}
        ]

    decision = metadata.get("baseline_decision", {}).get("decision")
    if decision != "promote":
        raise ValueError(
            f"Run nao aprovado pela politica de baseline: {decision or 'sem decisao'}."
        )

    audit_status = "not_available"
    leakage_path = settings.leakage_report_path if run_dir is None else (
        run_dir / settings.leakage_report_filename
    )
    if leakage_path.exists():
        audit_status = json.loads(leakage_path.read_text(encoding="utf-8"))["status"]
    registry = BaselineRegistry(settings)
    path = registry.promote(
        metadata,
        report_paths=report_paths,
        overwrite=args.overwrite,
        audit_status=audit_status,
        pipeline_path=pipeline_path,
    )
    if settings.database_tracking_enabled and run_dir is not None:
        persisted = PostgresTrainingHistoryRepository(
            replace(settings, promote_baseline=True)
        ).persist_if_available(run_dir)
        if not persisted:
            registry.rollback_promotion()
            raise RuntimeError("Promocao revertida porque o PostgreSQL nao foi atualizado.")
    registry.commit_promotion()
    uploaded = storage_sync.upload_artifacts(history_run_dir=run_dir)
    storage_sync.purge_local_artifacts(uploaded)
    if settings.storage_backend == "minio":
        location = settings.object_uri(
            settings.artifact_object_key(f"baseline/{path.name}")
        )
    else:
        location = str(path)
    print(f"Baseline oficial salvo em: {location}")


if __name__ == "__main__":
    main()
