"""Create fraud model training tracking tables.

Revision ID: 20260613_0001
Revises:
Create Date: 2026-06-13
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260613_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"


def upgrade() -> None:
    """Create normalized experiment tracking tables and indexes."""
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))

    op.create_table(
        "training_runs",
        sa.Column("run_id", sa.String(length=96), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Numeric(precision=16, scale=6), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("selected_threshold", sa.Numeric(precision=10, scale=8), nullable=False),
        sa.Column("threshold_strategy", sa.String(length=32), nullable=False),
        sa.Column("false_positive_cost", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("false_negative_cost", sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column("audit_status", sa.String(length=16), nullable=False),
        sa.Column("audit_warning_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("training_max_rows", sa.BigInteger(), nullable=True),
        sa.Column("train_rows", sa.BigInteger(), nullable=False),
        sa.Column("validation_rows", sa.BigInteger(), nullable=False),
        sa.Column("test_rows", sa.BigInteger(), nullable=False),
        sa.Column("train_positive_rate", sa.Float(), nullable=False),
        sa.Column("validation_positive_rate", sa.Float(), nullable=False),
        sa.Column("test_positive_rate", sa.Float(), nullable=False),
        sa.Column("strict_leakage_prevention", sa.Boolean(), nullable=False),
        sa.Column("pipeline_sha256", sa.String(length=64), nullable=False),
        sa.Column("run_directory", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("leakage_audit", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "audit_status IN ('pass', 'warning', 'fail')",
            name="ck_training_runs_audit_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_training_runs_completed_at",
        "training_runs",
        ["completed_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_training_runs_model_name",
        "training_runs",
        ["model_name"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_training_runs_audit_status",
        "training_runs",
        ["audit_status"],
        schema=SCHEMA,
    )

    op.create_table(
        "run_metrics",
        sa.Column(
            "run_id",
            sa.String(length=96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("split", sa.String(length=16), primary_key=True),
        sa.Column("threshold", sa.Numeric(precision=10, scale=8), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1", sa.Float(), nullable=False),
        sa.Column("fbeta", sa.Float(), nullable=False),
        sa.Column("pr_auc", sa.Float(), nullable=False),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("tp", sa.BigInteger(), nullable=False),
        sa.Column("fp", sa.BigInteger(), nullable=False),
        sa.Column("tn", sa.BigInteger(), nullable=False),
        sa.Column("fn", sa.BigInteger(), nullable=False),
        sa.Column("business_cost", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.CheckConstraint(
            "split IN ('validation', 'test')",
            name="ck_run_metrics_split",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_run_metrics_split_pr_auc",
        "run_metrics",
        ["split", "pr_auc"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_run_metrics_split_fbeta",
        "run_metrics",
        ["split", "fbeta"],
        schema=SCHEMA,
    )

    op.create_table(
        "threshold_evaluations",
        sa.Column(
            "run_id",
            sa.String(length=96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("split", sa.String(length=16), primary_key=True),
        sa.Column("threshold", sa.Numeric(precision=10, scale=8), primary_key=True),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1", sa.Float(), nullable=False),
        sa.Column("fbeta", sa.Float(), nullable=False),
        sa.Column("tp", sa.BigInteger(), nullable=False),
        sa.Column("fp", sa.BigInteger(), nullable=False),
        sa.Column("tn", sa.BigInteger(), nullable=False),
        sa.Column("fn", sa.BigInteger(), nullable=False),
        sa.Column("alerts", sa.BigInteger(), nullable=False),
        sa.Column("alert_rate", sa.Float(), nullable=False),
        sa.Column("business_cost", sa.Numeric(precision=20, scale=6), nullable=False),
        sa.Column("cost_per_record", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "split IN ('validation', 'test')",
            name="ck_threshold_evaluations_split",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_threshold_evaluations_cost",
        "threshold_evaluations",
        ["run_id", "split", "business_cost"],
        schema=SCHEMA,
    )

    op.create_table(
        "run_artifacts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "run_id",
            "artifact_type",
            "uri",
            name="uq_run_artifacts_run_type_uri",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_run_artifacts_run_id",
        "run_artifacts",
        ["run_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "baseline_promotions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("audit_status", sa.String(length=16), nullable=False),
        sa.Column("pipeline_sha256", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint(
            "audit_status IN ('pass', 'warning', 'fail', 'not_available')",
            name="ck_baseline_promotions_audit_status",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "uq_baseline_promotions_active",
        "baseline_promotions",
        ["is_active"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("is_active"),
    )


def downgrade() -> None:
    """Drop fraud tracking objects in dependency order."""
    op.drop_index(
        "uq_baseline_promotions_active",
        table_name="baseline_promotions",
        schema=SCHEMA,
    )
    op.drop_table("baseline_promotions", schema=SCHEMA)
    op.drop_index("ix_run_artifacts_run_id", table_name="run_artifacts", schema=SCHEMA)
    op.drop_table("run_artifacts", schema=SCHEMA)
    op.drop_index(
        "ix_threshold_evaluations_cost",
        table_name="threshold_evaluations",
        schema=SCHEMA,
    )
    op.drop_table("threshold_evaluations", schema=SCHEMA)
    op.drop_index("ix_run_metrics_split_fbeta", table_name="run_metrics", schema=SCHEMA)
    op.drop_index("ix_run_metrics_split_pr_auc", table_name="run_metrics", schema=SCHEMA)
    op.drop_table("run_metrics", schema=SCHEMA)
    op.drop_index("ix_training_runs_audit_status", table_name="training_runs", schema=SCHEMA)
    op.drop_index("ix_training_runs_model_name", table_name="training_runs", schema=SCHEMA)
    op.drop_index("ix_training_runs_completed_at", table_name="training_runs", schema=SCHEMA)
    op.drop_table("training_runs", schema=SCHEMA)
