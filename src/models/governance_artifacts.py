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


def write_model_review_report(
    path: Path,
    metadata: dict[str, Any],
    leakage_report: dict[str, Any],
    decision: dict[str, Any],
    target_audit: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    drift_report: dict[str, Any] | None,
    robustness_report: dict[str, Any] | None,
    walk_forward_report: dict[str, Any] | None,
    threshold_recommendations: dict[str, Any] | None,
) -> None:
    """Write a consolidated human-in-the-loop model review report."""
    lines = [
        "# Model Review Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Selected model: `{metadata['model_name']}`",
        f"- Final decision: `{decision['decision']}`",
        f"- Selected threshold: {metadata['threshold']:.6f}",
        f"- Leakage audit: `{leakage_report.get('status')}`",
        f"- Target audit: `{(target_audit or {}).get('status', 'not_available')}`",
        f"- Sampling audit: `{(sampling_audit or {}).get('status', 'not_available')}`",
        f"- Drift report: `{(drift_report or {}).get('status', 'not_available')}`",
        "",
        "## 2. Selected Model",
        "",
        f"- Selection engine: `{metadata.get('model_selection', {}).get('engine')}`",
        f"- Trials: {metadata.get('model_selection', {}).get('trial_count')}",
        f"- Dataset version: `{metadata.get('dataset_version')}`",
        f"- Feature set version: `{metadata.get('feature_set_version')}`",
        f"- Code version: `{metadata.get('code_version')}`",
        "",
        "## 3. Main Metrics",
        "",
        "| Split | PR-AUC | ROC-AUC | Precision | Recall | F-beta | Alert rate | Business cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test", "out_of_time"):
        metrics = metadata[f"{split}_metrics"]
        cost = metadata.get("operational_costs", {}).get(split)
        lines.append(
            f"| {split} | {metrics['pr_auc']:.6f} | {metrics['roc_auc']:.6f} | "
            f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | "
            f"{metrics['fbeta']:.6f} | {metrics['alert_rate']:.6f} | {cost:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Generalization",
            "",
            *_generalization_lines(metadata),
            "",
            "## 5. Baseline Decision",
            "",
            *[f"- Blocking: {reason}" for reason in decision.get("blocking_reasons", [])],
            *[f"- Warning: {warning}" for warning in decision.get("warnings", [])],
            *[f"- Recommendation: {item}" for item in decision.get("recommendations", [])],
            "",
            "## 6. Leakage Audit",
            "",
            *[f"- {warning}" for warning in leakage_report.get("warnings", [])],
            *[f"- Failure: {failure}" for failure in leakage_report.get("failures", [])],
            "",
            "## 7. Target Audit",
            "",
            *_section_status_lines(target_audit),
            "",
            "## Sampling Audit",
            "",
            *_sampling_audit_lines(sampling_audit),
            "",
            "## 9. Geographic Ablation",
            "",
            *_section_status_lines((robustness_report or {}).get("geo_ablation")),
            "",
            "## 10. Data Drift",
            "",
            *_section_status_lines(drift_report),
            "- See feature_stability_report.json and feature_stability_report.md for PSI by feature and keep/transform/remove recommendations.",
            "",
            "## 11. Threshold Analysis",
            "",
            "- Threshold ajusta o trade-off entre recall, precision, alert rate e custo; nao corrige score ruim, drift ou overfitting temporal.",
            *_threshold_lines(threshold_recommendations),
            "",
            "## 12. Walk-Forward Validation",
            "",
            *_section_status_lines(walk_forward_report),
            "",
            "## 13. Calibration",
            "",
            "- See calibration artifacts: calibration_report.csv, score_deciles.csv and calibration_curve.png.",
            "",
            "## 14. Score Deciles",
            "",
            "- See score_deciles.csv for score concentration and positive rate by decile.",
            "",
            "## 15. Conclusion",
            "",
            f"`{decision['decision']}`",
            "",
            "## 16. Next Actions",
            "",
            *[f"- {action}" for action in decision.get("next_actions", [])],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generalization_lines(metadata: dict[str, Any]) -> list[str]:
    validation = metadata["validation_metrics"]
    test = metadata["test_metrics"]
    oot = metadata["out_of_time_metrics"]
    validation_pr_auc = float(validation["pr_auc"])
    test_pr_auc = float(test["pr_auc"])
    oot_pr_auc = float(oot["pr_auc"])
    oot_drop = ((validation_pr_auc - oot_pr_auc) / validation_pr_auc) if validation_pr_auc > 0 else 0
    test_drop = ((validation_pr_auc - test_pr_auc) / validation_pr_auc) if validation_pr_auc > 0 else 0
    lines = [
        f"- Relative PR-AUC drop validation -> test: {test_drop:.2%}",
        f"- Relative PR-AUC drop validation -> out_of_time: {oot_drop:.2%}",
    ]
    if oot_drop > 0.5:
        lines.append(
            "- The model validates well but generalizes poorly out-of-time; treat validation performance as unstable."
        )
    return lines


def _section_status_lines(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return ["- Not available."]
    lines = [f"- Status: `{payload.get('status')}`"]
    lines.extend(f"- Note: {note}" for note in payload.get("notes", []))
    summary = payload.get("summary") or {}
    if summary:
        for key in (
            "best_fold",
            "worst_fold",
            "min_recall_fold",
            "min_pr_auc_fold",
            "last_fold_recall",
            "last_fold_pr_auc",
            "last_fold_penalty",
            "recall_drop_best_to_worst",
        ):
            if key in summary:
                lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(f"- Warning: {warning}" for warning in payload.get("warnings", []))
    lines.extend(f"- Failure: {failure}" for failure in payload.get("failures", []))
    return lines


def _sampling_audit_lines(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return ["- Not available."]
    before = payload.get("training_before_sampling") or {}
    after = payload.get("training_after_sampling") or {}
    lines = [
        f"- Status: `{payload.get('status')}`",
        f"- Dataset limitado: `{payload.get('training_limit_applied')}`",
        f"- Limite aplicado em: `{payload.get('training_limit_applied_stage')}`",
        f"- Positivos preservados: `{payload.get('positives_preserved_pct')}`",
        f"- Positivos antes/depois: {before.get('positive_count')} / {after.get('positive_count')}",
        f"- Range temporal preservado: `{payload.get('temporal_range_preserved')}`",
        f"- Amostragem sequencial: `{payload.get('sequential_training_limit_detected')}`",
        f"- Estrategia de negativos: `{payload.get('negative_sampling_strategy')}` por `{payload.get('negative_sampling_by')}`",
        f"- Rodada confiavel: `{payload.get('sampling_reliable')}`",
    ]
    lines.extend(f"- Warning: {item}" for item in payload.get("warnings", []))
    lines.extend(f"- Failure: {item}" for item in payload.get("failures", []))
    if payload.get("sequential_training_limit_detected"):
        lines.append(
            "- A rodada nao e confiavel para avaliacao temporal, pois o limite de linhas produziu vies temporal."
        )
    return lines


def _threshold_lines(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return ["- Not available."]
    lines = []
    for name, recommendation in payload.items():
        if recommendation is None:
            lines.append(f"- {name}: not available")
            continue
        lines.append(
            f"- {name}: threshold={recommendation.get('threshold')} | "
            f"split={recommendation.get('split')} | rationale={recommendation.get('rationale')}"
        )
    return lines


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
