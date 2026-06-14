"""Human-readable governance artifacts and integrity manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.versioning import sha256_file


def write_model_card(
    path: Path,
    metadata: dict[str, Any],
    leakage_report: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    """Write a concise model card for one training run."""
    dataset = metadata["dataset"]
    lines = [
        f"# Model Card: {metadata['model_name']}",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        "- Objective: fraud-risk triage for financial transactions.",
        "- Recommended use: prioritization and manual investigation.",
        "- Not recommended: automatic blocking without additional controls.",
        f"- Dataset version: `{metadata['dataset_version']}`",
        f"- Feature set version: `{metadata['feature_set_version']}`",
        f"- Code version: `{metadata['code_version']}`",
        f"- Model selection engine: `{metadata.get('model_selection', {}).get('engine', 'fixed')}`",
        f"- Model selection trials: {metadata.get('model_selection', {}).get('trial_count', 0)}",
        f"- Train rows: {dataset['train_rows']}",
        f"- Validation rows: {dataset['validation_rows']}",
        f"- Test rows: {dataset['test_rows']}",
        f"- Out-of-time rows: {dataset['out_of_time_rows']}",
        f"- Selected threshold: {metadata['threshold']:.6f}",
        f"- Threshold strategy: `{metadata['threshold_selection']['strategy']}`",
        f"- Leakage audit: `{leakage_report['status']}`",
        f"- Baseline decision: `{decision['decision']}`",
        "",
        "## Metrics",
        "",
        "| Split | PR-AUC | ROC-AUC | Precision | Recall | F-beta | Alert rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test", "out_of_time"):
        metrics = metadata[f"{split}_metrics"]
        lines.append(
            f"| {split} | {metrics['pr_auc']:.6f} | {metrics['roc_auc']:.6f} | "
            f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | "
            f"{metrics['fbeta']:.6f} | {metrics['alert_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            *[f"- {warning}" for warning in leakage_report.get("warnings", [])],
            "",
            "## Decision",
            "",
            *[f"- {reason}" for reason in decision["reasons"]],
            "",
            "## Limitations",
            "",
            "- Performance is dataset- and time-window-specific.",
            "- Feature coefficients express association, not causality.",
            "- Operational recall requires delayed fraud feedback.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, artifact_paths: list[Path]) -> dict[str, Any]:
    """Write an integrity manifest for every generated artifact except itself."""
    artifacts = [
        {
            "filename": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        }
        for artifact in sorted(artifact_paths)
        if artifact.exists() and artifact != path
    ]
    payload = {"artifact_count": len(artifacts), "artifacts": artifacts}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
