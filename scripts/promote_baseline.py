"""Promote the current trained model to the official baseline."""

from __future__ import annotations

import argparse
import json

import joblib

from src.config.settings import Settings
from src.models.baseline import BaselineRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Substitui o baseline oficial existente.")
    parser.add_argument("--run-id", help="Promove uma execucao especifica do historico.")
    return parser.parse_args()


def main() -> None:
    """Promote current pipeline and metadata artifacts."""
    args = parse_args()
    settings = Settings()
    pipeline_path = settings.pipeline_path
    if args.run_id:
        run_dir = settings.training_history_dir / args.run_id
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
            run_dir / settings.threshold_analysis_filename,
            run_dir / settings.leakage_report_filename,
        ]
    else:
        metadata = joblib.load(settings.metadata_path)
        report_paths = [settings.threshold_analysis_path, settings.leakage_report_path]

    audit_status = "not_available"
    leakage_path = report_paths[1]
    if leakage_path.exists():
        audit_status = json.loads(leakage_path.read_text(encoding="utf-8"))["status"]
    path = BaselineRegistry(settings).promote(
        metadata,
        report_paths=report_paths,
        overwrite=args.overwrite,
        audit_status=audit_status,
        pipeline_path=pipeline_path,
    )
    print(f"Baseline oficial salvo em: {path}")


if __name__ == "__main__":
    main()
