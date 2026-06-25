"""Objective candidate-to-baseline promotion policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.settings import Settings


class BaselineDecisionService:
    """Evaluate statistical, operational and governance promotion gates."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decide(
        self,
        metadata: dict[str, Any],
        leakage_report: dict[str, Any],
        required_artifacts: list[Path],
        target_audit: dict[str, Any] | None = None,
        drift_report: dict[str, Any] | None = None,
        robustness_report: dict[str, Any] | None = None,
        walk_forward_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return reject/candidate/pending_review/approved with explicit reasons."""
        blocking_reasons: list[str] = []
        warning_reasons: list[str] = []
        recommendations: list[str] = []
        audit_status = leakage_report["status"]
        oot = metadata.get("out_of_time_metrics")
        validation = metadata["validation_metrics"]
        test = metadata["test_metrics"]

        if audit_status == "fail":
            blocking_reasons.append("A auditoria de leakage possui falhas bloqueantes.")
        if oot is None:
            warning_reasons.append("Metricas out-of-time nao foram geradas.")
        else:
            validation_pr_auc = float(validation["pr_auc"])
            oot_pr_auc = float(oot["pr_auc"])
            relative_drop = (
                (validation_pr_auc - oot_pr_auc) / validation_pr_auc
                if validation_pr_auc > 0
                else 0.0
            )
            recall_drop = (
                (float(validation["recall"]) - float(oot["recall"])) / float(validation["recall"])
                if float(validation["recall"]) > 0
                else 0.0
            )
            if relative_drop > self.settings.promotion_max_oot_pr_auc_drop:
                blocking_reasons.append(
                    f"Queda relativa de PR-AUC OOT ({relative_drop:.2%}) excede o limite."
                )
            if recall_drop > self.settings.promotion_max_oot_pr_auc_drop:
                blocking_reasons.append(
                    f"Queda relativa de recall OOT ({recall_drop:.2%}) excede o limite."
                )
            if float(oot["recall"]) < self.settings.promotion_min_recall:
                blocking_reasons.append("Recall out-of-time abaixo do minimo operacional.")
            if float(oot["alert_rate"]) > self.settings.promotion_max_alert_rate:
                blocking_reasons.append("Alert rate out-of-time acima da capacidade operacional.")
            if abs(float(validation["recall"]) - float(test["recall"])) > 0.25:
                warning_reasons.append("Instabilidade relevante de recall entre validacao e teste.")

        if leakage_report["checks"].get("threshold_at_analysis_boundary"):
            warning_reasons.append("Threshold selecionado no limite da faixa analisada.")
        if leakage_report.get("warnings") and not self.settings.baseline_warning_justification:
            warning_reasons.append("Warnings da auditoria ainda nao possuem justificativa.")

        missing = [path.name for path in required_artifacts if not path.exists()]
        if missing:
            warning_reasons.append("Artefatos obrigatorios ausentes: " + ", ".join(sorted(missing)))

        self._apply_audit_gate("target audit", target_audit, blocking_reasons, warning_reasons)
        self._apply_audit_gate("drift report", drift_report, blocking_reasons, warning_reasons)
        geo_ablation = (robustness_report or {}).get("geo_ablation")
        self._apply_audit_gate("geo ablation", geo_ablation, blocking_reasons, warning_reasons)
        if walk_forward_report and walk_forward_report.get("status") != "disabled":
            self._apply_audit_gate("walk-forward validation", walk_forward_report, blocking_reasons, warning_reasons)

        baseline = self._load_current_baseline()
        if baseline:
            baseline_test = baseline.get("test_metrics", {})
            baseline_pr_auc = baseline_test.get("pr_auc")
            if baseline_pr_auc is not None and float(test["pr_auc"]) < float(baseline_pr_auc):
                blocking_reasons.append("PR-AUC de teste inferior ao baseline atual.")
            baseline_cost = self._cost_per_record(baseline_test)
            candidate_cost = self._cost_per_record(test)
            if (
                baseline_cost is not None
                and candidate_cost is not None
                and candidate_cost > baseline_cost * (1 + self.settings.promotion_max_cost_increase)
            ):
                blocking_reasons.append("Custo por registro superior ao limite do baseline.")

        if blocking_reasons:
            decision = "reject"
            recommendations.append("Nao promover; revisar generalizacao temporal e auditorias.")
        elif warning_reasons:
            decision = "pending_review"
            recommendations.append("Exigir aprovacao humana e justificativa antes de qualquer promocao.")
        elif self.settings.promote_baseline:
            decision = "approved"
            recommendations.append("Modelo apto para promocao conforme gates configurados.")
        else:
            decision = "candidate"
            recommendations.append("Candidato tecnicamente aprovado; promocao depende de aprovacao humana.")

        reasons = blocking_reasons + warning_reasons
        if not reasons:
            reasons = ["Todos os gates estatisticos, operacionais e de governanca foram aprovados."]

        return {
            "decision": decision,
            "reasons": reasons,
            "blocking_reasons": blocking_reasons,
            "warnings": warning_reasons,
            "recommendations": recommendations,
            "next_actions": self._next_actions(decision, blocking_reasons, warning_reasons),
            "warning_justification": self.settings.baseline_warning_justification,
            "policy": {
                "min_oot_recall": self.settings.promotion_min_recall,
                "max_oot_alert_rate": self.settings.promotion_max_alert_rate,
                "max_relative_oot_pr_auc_drop": self.settings.promotion_max_oot_pr_auc_drop,
                "max_baseline_cost_increase": self.settings.promotion_max_cost_increase,
                "human_approval_required_for_warnings": True,
            },
            "metrics_summary": {
                "validation_pr_auc": metadata["validation_metrics"].get("pr_auc"),
                "test_pr_auc": metadata["test_metrics"].get("pr_auc"),
                "out_of_time_pr_auc": (metadata.get("out_of_time_metrics") or {}).get("pr_auc"),
                "validation_recall": metadata["validation_metrics"].get("recall"),
                "out_of_time_recall": (metadata.get("out_of_time_metrics") or {}).get("recall"),
            },
        }

    @staticmethod
    def _apply_audit_gate(
        name: str,
        payload: dict[str, Any] | None,
        blocking_reasons: list[str],
        warning_reasons: list[str],
    ) -> None:
        if not payload:
            warning_reasons.append(f"{name} nao foi gerado.")
            return
        status = payload.get("status")
        if status == "fail":
            blocking_reasons.append(f"{name} possui falhas bloqueantes.")
        elif status == "warning":
            warning_reasons.append(f"{name} possui warnings pendentes.")

    @staticmethod
    def _next_actions(
        decision: str,
        blocking_reasons: list[str],
        warning_reasons: list[str],
    ) -> list[str]:
        if decision == "reject":
            return [
                "Investigar drift temporal, labels e dependencia geografica antes de novo treino.",
                *blocking_reasons,
            ]
        if decision == "pending_review":
            return [
                "Registrar justificativa humana para warnings ou ajustar configuracao/dados.",
                *warning_reasons,
            ]
        if decision == "candidate":
            return ["Revisar model_review_report.md e aprovar manualmente se o risco for aceitavel."]
        return ["Promocao permitida conforme politica atual; registrar aprovacao e monitoramento."]

    def _load_current_baseline(self) -> dict[str, Any] | None:
        path = self.settings.baseline_dir / self.settings.baseline_metadata_filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _cost_per_record(self, metrics: dict[str, Any]) -> float | None:
        required = {"fp", "fn", "tn", "tp"}
        if not required <= metrics.keys():
            return None
        count = sum(float(metrics[key]) for key in required)
        if count == 0:
            return None
        cost = (
            float(metrics["fp"]) * self.settings.false_positive_cost
            + float(metrics["fn"]) * self.settings.false_negative_cost
        )
        return cost / count
