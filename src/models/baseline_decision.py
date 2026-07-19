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
        sampling_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return reject/candidate/pending_review/approved with explicit reasons."""
        blocking_reasons: list[str] = []
        warning_reasons: list[str] = []
        recommendations: list[str] = []
        audit_status = leakage_report["status"]
        oot = metadata.get("out_of_time_metrics")
        validation = metadata["validation_metrics"]
        test = metadata["test_metrics"]

        if float(test.get("recall", 0.0)) < self.settings.promotion_min_recall:
            blocking_reasons.append("Recall de teste abaixo do minimo operacional.")
        test_positive_rate = (metadata.get("dataset") or {}).get("test_positive_rate")
        if (
            test_positive_rate is not None
            and float(test.get("pr_auc", 0.0))
            <= float(test_positive_rate) * self.settings.promotion_min_oot_pr_auc_lift
        ):
            blocking_reasons.append("PR-AUC de teste nao supera o baseline aleatorio exigido.")

        if audit_status == "fail":
            blocking_reasons.append("A auditoria de leakage possui falhas bloqueantes.")
        if oot is None:
            warning_reasons.append("Metricas out-of-time nao foram geradas.")
        else:
            validation_pr_auc = float(validation["pr_auc"])
            oot_pr_auc = float(oot["pr_auc"])
            if "tp" in oot and int(oot["tp"]) == 0:
                blocking_reasons.append("TP out-of-time igual a zero.")
            if float(oot.get("recall", 0.0)) == 0.0:
                blocking_reasons.append("Recall out-of-time igual a zero.")
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
            oot_positive_rate = (metadata.get("dataset") or {}).get("out_of_time_positive_rate")
            if oot_positive_rate is not None:
                random_pr_auc = float(oot_positive_rate)
                minimum_pr_auc = random_pr_auc * self.settings.promotion_min_oot_pr_auc_lift
                if oot_pr_auc <= minimum_pr_auc:
                    blocking_reasons.append(
                        "PR-AUC out-of-time nao supera o baseline aleatorio "
                        f"exigido ({oot_pr_auc:.6f} <= {minimum_pr_auc:.6f})."
                    )
            if abs(float(validation["recall"]) - float(test["recall"])) > 0.25:
                warning_reasons.append("Instabilidade relevante de recall entre validacao e teste.")

        if leakage_report["checks"].get("threshold_at_analysis_boundary"):
            warning_reasons.append("Threshold selecionado no limite da faixa analisada.")
        if leakage_report.get("warnings") and not self.settings.baseline_warning_justification:
            warning_reasons.append("Warnings da auditoria ainda nao possuem justificativa.")

        if target_audit:
            if target_audit.get("target_status") == "invalid":
                blocking_reasons.append("Target audit classificou o target como invalido.")
            if target_audit.get("unknown_labels_used_as_negative") or target_audit.get("unlabeled_as_negative"):
                blocking_reasons.append("Transacoes sem label foram usadas como classe 0.")

        missing = [path.name for path in required_artifacts if not path.exists()]
        if missing:
            warning_reasons.append("Artefatos obrigatorios ausentes: " + ", ".join(sorted(missing)))

        self._apply_audit_gate("target audit", target_audit, blocking_reasons, warning_reasons)
        if sampling_audit is not None:
            self._apply_audit_gate(
                "sampling audit", sampling_audit, blocking_reasons, warning_reasons
            )
        self._apply_audit_gate("drift report", drift_report, blocking_reasons, warning_reasons)
        geo_ablation = (robustness_report or {}).get("geo_ablation")
        self._apply_audit_gate("geo ablation", geo_ablation, blocking_reasons, warning_reasons)
        if walk_forward_report and walk_forward_report.get("status") != "disabled":
            self._apply_audit_gate("walk-forward validation", walk_forward_report, blocking_reasons, warning_reasons)
            self._apply_walk_forward_gate(walk_forward_report, blocking_reasons)

        baseline = self._load_current_baseline()
        if baseline:
            for split_name, candidate_metrics in (
                ("validation", validation),
                ("test", test),
                ("out_of_time", out_of_time),
            ):
                baseline_metrics = baseline.get(f"{split_name}_metrics", {})
                if not baseline_metrics or not candidate_metrics:
                    continue
                for metric in ("pr_auc", "recall", "precision"):
                    baseline_value = baseline_metrics.get(metric)
                    candidate_value = candidate_metrics.get(metric)
                    if (
                        baseline_value is not None
                        and candidate_value is not None
                        and float(candidate_value) < float(baseline_value)
                    ):
                        blocking_reasons.append(
                            f"{metric} de {split_name} inferior ao baseline atual."
                        )
                if (
                    baseline_metrics.get("fp") is not None
                    and candidate_metrics.get("fp") is not None
                    and float(candidate_metrics["fp"]) > float(baseline_metrics["fp"])
                ):
                    blocking_reasons.append(
                        f"Falsos positivos de {split_name} superiores ao baseline atual."
                    )
                baseline_cost = self._cost_per_record(baseline_metrics)
                candidate_cost = self._cost_per_record(candidate_metrics)
                if (
                    baseline_cost is not None
                    and candidate_cost is not None
                    and candidate_cost
                    > baseline_cost * (1 + self.settings.promotion_max_cost_increase)
                ):
                    blocking_reasons.append(
                        f"Custo por registro de {split_name} superior ao limite do baseline."
                    )

        if blocking_reasons:
            decision = "reject"
            recommendations.append("Nao promover; revisar generalizacao temporal e auditorias.")
        elif warning_reasons:
            decision = "pending_review"
            recommendations.append("Exigir aprovacao humana e justificativa antes de qualquer promocao.")
        elif self.settings.promote_baseline and self.settings.human_approval_confirmed:
            decision = "approved"
            recommendations.append("Modelo apto para promocao conforme gates configurados.")
        elif self.settings.promote_baseline:
            decision = "pending_review"
            warning_reasons.append("Aprovacao humana explicita ainda nao foi registrada.")
            recommendations.append("Registrar aprovacao humana antes da promocao.")
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
                "min_oot_pr_auc_lift_over_random": self.settings.promotion_min_oot_pr_auc_lift,
                "min_walk_forward_recall": self.settings.promotion_min_walk_forward_recall,
                "min_walk_forward_pr_auc_lift_over_random": self.settings.promotion_min_walk_forward_pr_auc_lift,
                "max_walk_forward_recall_drop": self.settings.promotion_max_walk_forward_recall_drop,
                "human_approval_required_for_warnings": True,
            },
            "metrics_summary": {
                "validation_pr_auc": metadata["validation_metrics"].get("pr_auc"),
                "test_pr_auc": metadata["test_metrics"].get("pr_auc"),
                "out_of_time_pr_auc": (metadata.get("out_of_time_metrics") or {}).get("pr_auc"),
                "validation_recall": metadata["validation_metrics"].get("recall"),
                "out_of_time_recall": (metadata.get("out_of_time_metrics") or {}).get("recall"),
                "last_fold_recall": (walk_forward_report or {}).get("summary", {}).get("last_fold_recall"),
                "min_fold_recall": (walk_forward_report or {}).get("summary", {}).get("min_recall_fold"),
                "last_fold_pr_auc": (walk_forward_report or {}).get("summary", {}).get("last_fold_pr_auc"),
                "min_fold_pr_auc": (walk_forward_report or {}).get("summary", {}).get("min_pr_auc_fold"),
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
    def _apply_walk_forward_gate(
        payload: dict[str, Any],
        blocking_reasons: list[str],
    ) -> None:
        summary = payload.get("summary") or {}
        if payload.get("status") == "fail":
            blocking_reasons.append("Walk-forward falhou em gates temporais criticos.")
        for reason in summary.get("failure_reasons", []):
            if reason not in blocking_reasons:
                blocking_reasons.append(reason)

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
