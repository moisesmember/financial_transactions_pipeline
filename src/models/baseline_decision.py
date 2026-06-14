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
    ) -> dict[str, Any]:
        """Return promote, keep_candidate or reject with explicit reasons."""
        rejection_reasons: list[str] = []
        pending_reasons: list[str] = []
        audit_status = leakage_report["status"]
        oot = metadata.get("out_of_time_metrics")
        test = metadata["test_metrics"]

        if audit_status == "fail":
            rejection_reasons.append("A auditoria de leakage possui falhas bloqueantes.")
        if oot is None:
            pending_reasons.append("Metricas out-of-time nao foram geradas.")
        else:
            test_pr_auc = float(test["pr_auc"])
            oot_pr_auc = float(oot["pr_auc"])
            relative_drop = (
                (test_pr_auc - oot_pr_auc) / test_pr_auc if test_pr_auc > 0 else 0.0
            )
            if relative_drop > self.settings.promotion_max_oot_pr_auc_drop:
                rejection_reasons.append(
                    f"Queda relativa de PR-AUC OOT ({relative_drop:.2%}) excede o limite."
                )
            if float(oot["recall"]) < self.settings.promotion_min_recall:
                rejection_reasons.append("Recall out-of-time abaixo do minimo operacional.")
            if float(oot["alert_rate"]) > self.settings.promotion_max_alert_rate:
                rejection_reasons.append("Alert rate out-of-time acima da capacidade operacional.")

        if leakage_report["checks"].get("threshold_at_analysis_boundary"):
            pending_reasons.append("Threshold selecionado no limite da faixa analisada.")
        if leakage_report.get("warnings") and not self.settings.baseline_warning_justification:
            pending_reasons.append("Warnings da auditoria ainda nao possuem justificativa.")

        missing = [path.name for path in required_artifacts if not path.exists()]
        if missing:
            pending_reasons.append("Artefatos obrigatorios ausentes: " + ", ".join(sorted(missing)))

        baseline = self._load_current_baseline()
        if baseline:
            baseline_test = baseline.get("test_metrics", {})
            baseline_pr_auc = baseline_test.get("pr_auc")
            if baseline_pr_auc is not None and float(test["pr_auc"]) < float(baseline_pr_auc):
                rejection_reasons.append("PR-AUC de teste inferior ao baseline atual.")
            baseline_cost = self._cost_per_record(baseline_test)
            candidate_cost = self._cost_per_record(test)
            if (
                baseline_cost is not None
                and candidate_cost is not None
                and candidate_cost > baseline_cost * (1 + self.settings.promotion_max_cost_increase)
            ):
                rejection_reasons.append("Custo por registro superior ao limite do baseline.")

        if rejection_reasons:
            decision = "reject"
            reasons = rejection_reasons + pending_reasons
        elif pending_reasons:
            decision = "keep_candidate"
            reasons = pending_reasons
        else:
            decision = "promote"
            reasons = ["Todos os gates estatisticos, operacionais e de governanca foram aprovados."]

        return {
            "decision": decision,
            "reasons": reasons,
            "warning_justification": self.settings.baseline_warning_justification,
            "policy": {
                "min_oot_recall": self.settings.promotion_min_recall,
                "max_oot_alert_rate": self.settings.promotion_max_alert_rate,
                "max_relative_oot_pr_auc_drop": self.settings.promotion_max_oot_pr_auc_drop,
                "max_baseline_cost_increase": self.settings.promotion_max_cost_increase,
            },
        }

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
