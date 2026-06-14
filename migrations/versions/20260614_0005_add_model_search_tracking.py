"""Add Optuna model search and external benchmark tracking.

Revision ID: 20260614_0005
Revises: 20260614_0004
Create Date: 2026-06-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260614_0005"
down_revision: str | None = "20260614_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"
FACT_VIEW = "fact_model_runs"
PREVIOUS_FACT_VIEW = "fact_model_runs_v4"


def upgrade() -> None:
    """Create normalized model-selection and benchmark result tables."""
    op.execute(
        sa.text(
            f"ALTER VIEW {SCHEMA}.{FACT_VIEW} RENAME TO {PREVIOUS_FACT_VIEW}"
        )
    )
    op.add_column(
        "training_runs",
        sa.Column("model_selection_engine", sa.String(32), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "training_runs",
        sa.Column("model_selection_objective", sa.String(64), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "training_runs",
        sa.Column("model_selection_trial_count", sa.Integer(), nullable=True),
        schema=SCHEMA,
    )

    op.create_table(
        "model_search_trials",
        sa.Column(
            "run_id",
            sa.String(96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("trial_number", sa.Integer(), primary_key=True),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column("model_params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("validation_pr_auc", sa.Float(), nullable=True),
        sa.Column("selected_threshold", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("alert_rate", sa.Float(), nullable=True),
        sa.Column("business_cost", sa.Numeric(20, 6), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_model_search_trials_pr_auc",
        "model_search_trials",
        ["run_id", "validation_pr_auc"],
        schema=SCHEMA,
    )
    op.execute(
        sa.text(
            f"""
            CREATE VIEW {SCHEMA}.{FACT_VIEW} AS
            SELECT
                previous.*,
                runs.model_selection_engine,
                runs.model_selection_objective,
                runs.model_selection_trial_count
            FROM {SCHEMA}.{PREVIOUS_FACT_VIEW} previous
            JOIN {SCHEMA}.training_runs runs USING (run_id)
            """
        )
    )

    op.create_table(
        "external_benchmark_results",
        sa.Column(
            "run_id",
            sa.String(96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("backend", sa.String(32), primary_key=True),
        sa.Column("split", sa.String(24), primary_key=True),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("framework_model", sa.Text(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("fbeta", sa.Float(), nullable=True),
        sa.Column("pr_auc", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("tp", sa.BigInteger(), nullable=True),
        sa.Column("fp", sa.BigInteger(), nullable=True),
        sa.Column("tn", sa.BigInteger(), nullable=True),
        sa.Column("fn", sa.BigInteger(), nullable=True),
        sa.Column("alerts", sa.BigInteger(), nullable=True),
        sa.Column("alert_rate", sa.Float(), nullable=True),
        sa.Column("business_cost", sa.Numeric(20, 6), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_external_benchmark_results_pr_auc",
        "external_benchmark_results",
        ["run_id", "split", "pr_auc"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Remove model-selection and external benchmark tracking."""
    op.execute(sa.text(f"DROP VIEW IF EXISTS {SCHEMA}.{FACT_VIEW}"))
    op.drop_index(
        "ix_external_benchmark_results_pr_auc",
        table_name="external_benchmark_results",
        schema=SCHEMA,
    )
    op.drop_table("external_benchmark_results", schema=SCHEMA)
    op.drop_index(
        "ix_model_search_trials_pr_auc",
        table_name="model_search_trials",
        schema=SCHEMA,
    )
    op.drop_table("model_search_trials", schema=SCHEMA)
    op.drop_column("training_runs", "model_selection_trial_count", schema=SCHEMA)
    op.drop_column("training_runs", "model_selection_objective", schema=SCHEMA)
    op.drop_column("training_runs", "model_selection_engine", schema=SCHEMA)
    op.execute(
        sa.text(
            f"ALTER VIEW {SCHEMA}.{PREVIOUS_FACT_VIEW} RENAME TO {FACT_VIEW}"
        )
    )
