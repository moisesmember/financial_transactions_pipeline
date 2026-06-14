"""Create consolidated model run fact view.

Revision ID: 20260614_0002
Revises: 20260613_0001
Create Date: 2026-06-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260614_0002"
down_revision: str | None = "20260613_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"
VIEW = "fact_model_runs"


def upgrade() -> None:
    """Create one analytical row per training run."""
    op.execute(
        sa.text(
            f"""
            CREATE VIEW {SCHEMA}.{VIEW} AS
            WITH metrics AS (
                SELECT
                    run_id,
                    MAX(threshold) FILTER (WHERE split = 'validation') AS validation_threshold,
                    MAX(precision) FILTER (WHERE split = 'validation') AS validation_precision,
                    MAX(recall) FILTER (WHERE split = 'validation') AS validation_recall,
                    MAX(f1) FILTER (WHERE split = 'validation') AS validation_f1,
                    MAX(fbeta) FILTER (WHERE split = 'validation') AS validation_fbeta,
                    MAX(pr_auc) FILTER (WHERE split = 'validation') AS validation_pr_auc,
                    MAX(roc_auc) FILTER (WHERE split = 'validation') AS validation_roc_auc,
                    MAX(tp) FILTER (WHERE split = 'validation') AS validation_tp,
                    MAX(fp) FILTER (WHERE split = 'validation') AS validation_fp,
                    MAX(tn) FILTER (WHERE split = 'validation') AS validation_tn,
                    MAX(fn) FILTER (WHERE split = 'validation') AS validation_fn,
                    MAX(business_cost) FILTER (WHERE split = 'validation') AS validation_business_cost,
                    MAX(threshold) FILTER (WHERE split = 'test') AS test_threshold,
                    MAX(precision) FILTER (WHERE split = 'test') AS test_precision,
                    MAX(recall) FILTER (WHERE split = 'test') AS test_recall,
                    MAX(f1) FILTER (WHERE split = 'test') AS test_f1,
                    MAX(fbeta) FILTER (WHERE split = 'test') AS test_fbeta,
                    MAX(pr_auc) FILTER (WHERE split = 'test') AS test_pr_auc,
                    MAX(roc_auc) FILTER (WHERE split = 'test') AS test_roc_auc,
                    MAX(tp) FILTER (WHERE split = 'test') AS test_tp,
                    MAX(fp) FILTER (WHERE split = 'test') AS test_fp,
                    MAX(tn) FILTER (WHERE split = 'test') AS test_tn,
                    MAX(fn) FILTER (WHERE split = 'test') AS test_fn,
                    MAX(business_cost) FILTER (WHERE split = 'test') AS test_business_cost
                FROM {SCHEMA}.run_metrics
                GROUP BY run_id
            ),
            thresholds AS (
                SELECT
                    run_id,
                    COUNT(*) AS threshold_evaluation_count,
                    MIN(threshold) AS threshold_min,
                    MAX(threshold) AS threshold_max,
                    jsonb_agg(
                        jsonb_build_object(
                            'split', split,
                            'threshold', threshold,
                            'precision', precision,
                            'recall', recall,
                            'f1', f1,
                            'fbeta', fbeta,
                            'tp', tp,
                            'fp', fp,
                            'tn', tn,
                            'fn', fn,
                            'alerts', alerts,
                            'alert_rate', alert_rate,
                            'business_cost', business_cost,
                            'cost_per_record', cost_per_record
                        )
                        ORDER BY split, threshold
                    ) AS threshold_evaluations
                FROM {SCHEMA}.threshold_evaluations
                GROUP BY run_id
            ),
            artifacts AS (
                SELECT
                    run_id,
                    COUNT(*) AS artifact_count,
                    COALESCE(SUM(size_bytes), 0) AS artifact_total_size_bytes,
                    jsonb_agg(
                        jsonb_build_object(
                            'artifact_type', artifact_type,
                            'uri', uri,
                            'sha256', sha256,
                            'size_bytes', size_bytes,
                            'created_at', created_at
                        )
                        ORDER BY artifact_type, uri
                    ) AS artifacts
                FROM {SCHEMA}.run_artifacts
                GROUP BY run_id
            ),
            promotions AS (
                SELECT
                    run_id,
                    COUNT(*) AS baseline_promotion_count,
                    BOOL_OR(is_active) AS is_active_baseline,
                    MAX(promoted_at) AS last_promoted_at,
                    jsonb_agg(
                        jsonb_build_object(
                            'promotion_id', id,
                            'promoted_at', promoted_at,
                            'audit_status', audit_status,
                            'pipeline_sha256', pipeline_sha256,
                            'is_active', is_active,
                            'metadata', metadata
                        )
                        ORDER BY promoted_at DESC
                    ) AS baseline_promotions
                FROM {SCHEMA}.baseline_promotions
                WHERE run_id IS NOT NULL
                GROUP BY run_id
            ),
            fact AS (
                SELECT
                    tr.run_id,
                    tr.started_at,
                    tr.completed_at,
                    tr.duration_seconds,
                    tr.model_name,
                    tr.selected_threshold,
                    tr.threshold_strategy,
                    tr.false_positive_cost,
                    tr.false_negative_cost,
                    tr.audit_status,
                    tr.audit_warning_count,
                    tr.training_max_rows,
                    tr.train_rows,
                    tr.validation_rows,
                    tr.test_rows,
                    tr.train_positive_rate,
                    tr.validation_positive_rate,
                    tr.test_positive_rate,
                    tr.strict_leakage_prevention,
                    tr.pipeline_sha256,
                    tr.run_directory,
                    tr.created_at,
                    tr.metadata,
                    tr.leakage_audit,
                    m.validation_threshold,
                    m.validation_precision,
                    m.validation_recall,
                    m.validation_f1,
                    m.validation_fbeta,
                    m.validation_pr_auc,
                    m.validation_roc_auc,
                    m.validation_tp,
                    m.validation_fp,
                    m.validation_tn,
                    m.validation_fn,
                    m.validation_business_cost,
                    m.test_threshold,
                    m.test_precision,
                    m.test_recall,
                    m.test_f1,
                    m.test_fbeta,
                    m.test_pr_auc,
                    m.test_roc_auc,
                    m.test_tp,
                    m.test_fp,
                    m.test_tn,
                    m.test_fn,
                    m.test_business_cost,
                    COALESCE(t.threshold_evaluation_count, 0) AS threshold_evaluation_count,
                    t.threshold_min,
                    t.threshold_max,
                    COALESCE(t.threshold_evaluations, '[]'::jsonb) AS threshold_evaluations,
                    COALESCE(a.artifact_count, 0) AS artifact_count,
                    COALESCE(a.artifact_total_size_bytes, 0) AS artifact_total_size_bytes,
                    COALESCE(a.artifacts, '[]'::jsonb) AS artifacts,
                    COALESCE(p.baseline_promotion_count, 0) AS baseline_promotion_count,
                    COALESCE(p.is_active_baseline, false) AS is_active_baseline,
                    p.last_promoted_at,
                    COALESCE(p.baseline_promotions, '[]'::jsonb) AS baseline_promotions
                FROM {SCHEMA}.training_runs tr
                LEFT JOIN metrics m ON m.run_id = tr.run_id
                LEFT JOIN thresholds t ON t.run_id = tr.run_id
                LEFT JOIN artifacts a ON a.run_id = tr.run_id
                LEFT JOIN promotions p ON p.run_id = tr.run_id
            )
            SELECT
                fact.*,
                DENSE_RANK() OVER (
                    ORDER BY fact.test_pr_auc DESC NULLS LAST
                ) AS test_pr_auc_rank,
                DENSE_RANK() OVER (
                    ORDER BY fact.test_fbeta DESC NULLS LAST
                ) AS test_fbeta_rank,
                DENSE_RANK() OVER (
                    ORDER BY fact.test_business_cost ASC NULLS LAST
                ) AS test_business_cost_rank,
                DENSE_RANK() OVER (
                    PARTITION BY fact.model_name
                    ORDER BY fact.test_pr_auc DESC NULLS LAST
                ) AS model_test_pr_auc_rank
            FROM fact
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            COMMENT ON VIEW {SCHEMA}.{VIEW} IS
            'One analytical fact row per training run, including metrics, costs, audit, thresholds, artifacts, baseline promotions, and performance ranks.'
            """
        )
    )


def downgrade() -> None:
    """Drop the consolidated fact view."""
    op.execute(sa.text(f"DROP VIEW IF EXISTS {SCHEMA}.{VIEW}"))
