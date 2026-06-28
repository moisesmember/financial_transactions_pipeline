"""Expand model governance, OOT validation and monitoring schema.

Revision ID: 20260614_0003
Revises: 20260614_0002
Create Date: 2026-06-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260614_0003"
down_revision: str | None = "20260614_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"
VIEW = "fact_model_runs"


def upgrade() -> None:
    """Add governed lifecycle, OOT, detailed audits and monitoring tables."""
    op.execute(sa.text(f"DROP VIEW IF EXISTS {SCHEMA}.{VIEW}"))

    for column in (
        sa.Column("status", sa.String(16), nullable=False, server_default="completed"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("audit_failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("out_of_time_rows", sa.BigInteger(), nullable=True),
        sa.Column("out_of_time_positive_rate", sa.Float(), nullable=True),
        sa.Column("dataset_version", sa.String(128), nullable=True),
        sa.Column("dataset_sha256", sa.String(64), nullable=True),
        sa.Column("feature_set_version", sa.String(64), nullable=True),
        sa.Column("code_version", sa.String(128), nullable=True),
        sa.Column("experiment_fingerprint", sa.String(64), nullable=True),
        sa.Column("experiment_group", sa.String(64), nullable=True),
        sa.Column("features_removed", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("promotion_decision", sa.String(24), nullable=True),
        sa.Column("promotion_reason", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    ):
        op.add_column("training_runs", column, schema=SCHEMA)
    op.create_check_constraint(
        "ck_training_runs_status",
        "training_runs",
        "status IN ('running', 'completed', 'failed', 'rejected', 'promoted')",
        schema=SCHEMA,
    )
    op.create_index(
        "ix_training_runs_experiment_fingerprint",
        "training_runs",
        ["experiment_fingerprint"],
        schema=SCHEMA,
    )

    op.drop_constraint("ck_run_metrics_split", "run_metrics", schema=SCHEMA, type_="check")
    op.create_check_constraint(
        "ck_run_metrics_split",
        "run_metrics",
        "split IN ('train', 'validation', 'test', 'out_of_time')",
        schema=SCHEMA,
    )
    op.add_column("run_metrics", sa.Column("alerts", sa.BigInteger(), nullable=True), schema=SCHEMA)
    op.add_column("run_metrics", sa.Column("alert_rate", sa.Float(), nullable=True), schema=SCHEMA)
    op.add_column(
        "run_metrics",
        sa.Column("cost_per_record", sa.Float(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "run_metrics",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema=SCHEMA,
    )

    op.drop_constraint(
        "threshold_evaluations_pkey",
        "threshold_evaluations",
        schema=SCHEMA,
        type_="primary",
    )
    op.drop_constraint(
        "ck_threshold_evaluations_split",
        "threshold_evaluations",
        schema=SCHEMA,
        type_="check",
    )
    op.add_column(
        "threshold_evaluations",
        sa.Column("scenario_name", sa.String(64), nullable=False, server_default="primary"),
        schema=SCHEMA,
    )
    op.add_column(
        "threshold_evaluations",
        sa.Column("false_positive_cost", sa.Numeric(18, 6), nullable=False, server_default="0"),
        schema=SCHEMA,
    )
    op.add_column(
        "threshold_evaluations",
        sa.Column("false_negative_cost", sa.Numeric(18, 6), nullable=False, server_default="0"),
        schema=SCHEMA,
    )
    op.add_column(
        "threshold_evaluations",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_threshold_evaluations_split",
        "threshold_evaluations",
        "split IN ('validation', 'test', 'out_of_time')",
        schema=SCHEMA,
    )
    op.create_primary_key(
        "pk_threshold_evaluations",
        "threshold_evaluations",
        ["run_id", "scenario_name", "split", "threshold"],
        schema=SCHEMA,
    )

    op.create_table(
        "leakage_audit_checks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("check_name", sa.String(128), nullable=False),
        sa.Column("check_result", sa.String(16), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("run_id", "check_name", name="uq_leakage_check_run_name"),
        schema=SCHEMA,
    )
    op.create_table(
        "model_features",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("absolute_importance", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(16), nullable=True),
        sa.Column("odds_ratio", sa.Float(), nullable=True),
        sa.Column("feature_group", sa.String(32), nullable=True),
        sa.Column("is_geo_feature", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_temporal_feature", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_behavioral_feature", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_risk_feature", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("run_id", "feature_name", name="uq_model_feature_run_name"),
        schema=SCHEMA,
    )
    for column in (
        sa.Column("previous_baseline_run_id", sa.String(96), nullable=True),
        sa.Column("promoted_by", sa.String(128), nullable=True),
        sa.Column("decision", sa.String(24), nullable=True),
        sa.Column("decision_reason", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("approval_status", sa.String(24), nullable=True),
        sa.Column("rollback_available", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    ):
        op.add_column("baseline_promotions", column, schema=SCHEMA)

    _create_monitoring_tables()
    _create_fact_view()


def downgrade() -> None:
    """Remove governance additions and restore the previous fact view shape."""
    op.execute(sa.text(f"DROP VIEW IF EXISTS {SCHEMA}.{VIEW}"))
    for table in ("drift_metrics", "operational_feedback", "model_predictions"):
        op.drop_table(table, schema=SCHEMA)
    for column in (
        "created_at",
        "rollback_available",
        "approval_status",
        "decision_reason",
        "decision",
        "promoted_by",
        "previous_baseline_run_id",
    ):
        op.drop_column("baseline_promotions", column, schema=SCHEMA)
    op.drop_table("model_features", schema=SCHEMA)
    op.drop_table("leakage_audit_checks", schema=SCHEMA)
    op.drop_constraint("pk_threshold_evaluations", "threshold_evaluations", schema=SCHEMA)
    op.drop_constraint(
        "ck_threshold_evaluations_split",
        "threshold_evaluations",
        schema=SCHEMA,
        type_="check",
    )
    for column in ("created_at", "false_negative_cost", "false_positive_cost", "scenario_name"):
        op.drop_column("threshold_evaluations", column, schema=SCHEMA)
    op.create_primary_key(
        "threshold_evaluations_pkey",
        "threshold_evaluations",
        ["run_id", "split", "threshold"],
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_threshold_evaluations_split",
        "threshold_evaluations",
        "split IN ('validation', 'test')",
        schema=SCHEMA,
    )
    op.drop_constraint("ck_run_metrics_split", "run_metrics", schema=SCHEMA, type_="check")
    for column in ("created_at", "cost_per_record", "alert_rate", "alerts"):
        op.drop_column("run_metrics", column, schema=SCHEMA)
    op.create_check_constraint(
        "ck_run_metrics_split",
        "run_metrics",
        "split IN ('validation', 'test')",
        schema=SCHEMA,
    )
    op.drop_index("ix_training_runs_experiment_fingerprint", table_name="training_runs", schema=SCHEMA)
    op.drop_constraint("ck_training_runs_status", "training_runs", schema=SCHEMA, type_="check")
    for column in (
        "promotion_reason",
        "promotion_decision",
        "features_removed",
        "experiment_group",
        "experiment_fingerprint",
        "code_version",
        "feature_set_version",
        "dataset_sha256",
        "dataset_version",
        "out_of_time_positive_rate",
        "out_of_time_rows",
        "audit_failure_count",
        "failure_reason",
        "status",
    ):
        op.drop_column("training_runs", column, schema=SCHEMA)
    _create_fact_view(expanded=False)


def _create_monitoring_tables() -> None:
    op.create_table(
        "model_predictions",
        sa.Column("prediction_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(96), nullable=True),
        sa.Column("transaction_id", sa.String(128), nullable=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("is_alert", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("feature_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema=SCHEMA,
    )
    op.create_table(
        "operational_feedback",
        sa.Column("feedback_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("confirmed_fraud", sa.Boolean(), nullable=False),
        sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(64), nullable=True),
        schema=SCHEMA,
    )
    op.create_table(
        "drift_metrics",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(96), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metric_name", sa.String(128), nullable=False),
        sa.Column("feature_name", sa.String(256), nullable=True),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("severity", sa.String(16), nullable=True),
        schema=SCHEMA,
    )


def _create_fact_view(expanded: bool = True) -> None:
    if not expanded:
        op.execute(
            sa.text(
                f"""
                CREATE VIEW {SCHEMA}.{VIEW} AS
                WITH metrics AS (
                    SELECT run_id,
                        MAX(pr_auc) FILTER (WHERE split = 'validation') AS validation_pr_auc,
                        MAX(fbeta) FILTER (WHERE split = 'validation') AS validation_fbeta,
                        MAX(pr_auc) FILTER (WHERE split = 'test') AS test_pr_auc,
                        MAX(fbeta) FILTER (WHERE split = 'test') AS test_fbeta,
                        MAX(business_cost) FILTER (WHERE split = 'test') AS test_business_cost
                    FROM {SCHEMA}.run_metrics GROUP BY run_id
                )
                SELECT tr.*,
                    m.validation_pr_auc, m.validation_fbeta, m.test_pr_auc, m.test_fbeta,
                    m.test_business_cost,
                    DENSE_RANK() OVER (ORDER BY m.test_pr_auc DESC NULLS LAST) AS test_pr_auc_rank,
                    DENSE_RANK() OVER (ORDER BY m.test_fbeta DESC NULLS LAST) AS test_fbeta_rank,
                    DENSE_RANK() OVER (
                        ORDER BY m.test_business_cost ASC NULLS LAST
                    ) AS test_business_cost_rank,
                    DENSE_RANK() OVER (
                        PARTITION BY tr.model_name ORDER BY m.test_pr_auc DESC NULLS LAST
                    ) AS model_test_pr_auc_rank
                FROM {SCHEMA}.training_runs tr
                LEFT JOIN metrics m ON m.run_id = tr.run_id
                """
            )
        )
        return
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
                    MAX(alert_rate) FILTER (WHERE split = 'validation') AS validation_alert_rate,
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
                    MAX(alert_rate) FILTER (WHERE split = 'test') AS test_alert_rate,
                    MAX(business_cost) FILTER (WHERE split = 'test') AS test_business_cost,
                    MAX(threshold) FILTER (WHERE split = 'out_of_time') AS out_of_time_threshold,
                    MAX(precision) FILTER (WHERE split = 'out_of_time') AS out_of_time_precision,
                    MAX(recall) FILTER (WHERE split = 'out_of_time') AS out_of_time_recall,
                    MAX(f1) FILTER (WHERE split = 'out_of_time') AS out_of_time_f1,
                    MAX(fbeta) FILTER (WHERE split = 'out_of_time') AS out_of_time_fbeta,
                    MAX(pr_auc) FILTER (WHERE split = 'out_of_time') AS out_of_time_pr_auc,
                    MAX(roc_auc) FILTER (WHERE split = 'out_of_time') AS out_of_time_roc_auc,
                    MAX(tp) FILTER (WHERE split = 'out_of_time') AS out_of_time_tp,
                    MAX(fp) FILTER (WHERE split = 'out_of_time') AS out_of_time_fp,
                    MAX(tn) FILTER (WHERE split = 'out_of_time') AS out_of_time_tn,
                    MAX(fn) FILTER (WHERE split = 'out_of_time') AS out_of_time_fn,
                    MAX(alert_rate) FILTER (WHERE split = 'out_of_time') AS out_of_time_alert_rate,
                    MAX(business_cost) FILTER (WHERE split = 'out_of_time') AS out_of_time_business_cost
                FROM {SCHEMA}.run_metrics
                GROUP BY run_id
            ),
            thresholds AS (
                SELECT run_id, COUNT(*) AS threshold_evaluation_count,
                    jsonb_agg(to_jsonb(te) - 'run_id' ORDER BY scenario_name, split, threshold)
                    AS threshold_evaluations
                FROM {SCHEMA}.threshold_evaluations te GROUP BY run_id
            ),
            artifacts AS (
                SELECT run_id, COUNT(*) AS artifact_count,
                    COALESCE(SUM(size_bytes), 0) AS artifact_total_size_bytes,
                    jsonb_agg(to_jsonb(a) - 'id' - 'run_id' ORDER BY artifact_type, uri) AS artifacts
                FROM {SCHEMA}.run_artifacts a GROUP BY run_id
            ),
            promotions AS (
                SELECT run_id, COUNT(*) AS baseline_promotion_count,
                    BOOL_OR(is_active) AS is_active_baseline,
                    MAX(promoted_at) AS last_promoted_at,
                    jsonb_agg(to_jsonb(p) - 'run_id' ORDER BY promoted_at DESC) AS baseline_promotions
                FROM {SCHEMA}.baseline_promotions p WHERE run_id IS NOT NULL GROUP BY run_id
            )
            SELECT tr.*,
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
                m.validation_alert_rate,
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
                m.test_alert_rate,
                m.test_business_cost,
                m.out_of_time_threshold,
                m.out_of_time_precision,
                m.out_of_time_recall,
                m.out_of_time_f1,
                m.out_of_time_fbeta,
                m.out_of_time_pr_auc,
                m.out_of_time_roc_auc,
                m.out_of_time_tp,
                m.out_of_time_fp,
                m.out_of_time_tn,
                m.out_of_time_fn,
                m.out_of_time_alert_rate,
                m.out_of_time_business_cost,
                COALESCE(t.threshold_evaluation_count, 0) AS threshold_evaluation_count,
                COALESCE(t.threshold_evaluations, '[]'::jsonb) AS threshold_evaluations,
                COALESCE(a.artifact_count, 0) AS artifact_count,
                COALESCE(a.artifact_total_size_bytes, 0) AS artifact_total_size_bytes,
                COALESCE(a.artifacts, '[]'::jsonb) AS artifacts,
                COALESCE(p.baseline_promotion_count, 0) AS baseline_promotion_count,
                COALESCE(p.is_active_baseline, false) AS is_active_baseline,
                p.last_promoted_at,
                COALESCE(p.baseline_promotions, '[]'::jsonb) AS baseline_promotions,
                DENSE_RANK() OVER (ORDER BY m.test_pr_auc DESC NULLS LAST) AS test_pr_auc_rank,
                DENSE_RANK() OVER (ORDER BY m.test_fbeta DESC NULLS LAST) AS test_fbeta_rank,
                DENSE_RANK() OVER (ORDER BY m.test_business_cost ASC NULLS LAST) AS test_business_cost_rank,
                DENSE_RANK() OVER (
                    PARTITION BY tr.model_name ORDER BY m.test_pr_auc DESC NULLS LAST
                ) AS model_test_pr_auc_rank
            FROM {SCHEMA}.training_runs tr
            LEFT JOIN metrics m ON m.run_id = tr.run_id
            LEFT JOIN thresholds t ON t.run_id = tr.run_id
            LEFT JOIN artifacts a ON a.run_id = tr.run_id
            LEFT JOIN promotions p ON p.run_id = tr.run_id
            """
        )
    )
