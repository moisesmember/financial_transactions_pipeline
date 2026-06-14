"""Add geographic robustness experiment tracking.

Revision ID: 20260614_0004
Revises: 20260614_0003
Create Date: 2026-06-14
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260614_0004"
down_revision: str | None = "20260614_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"


def upgrade() -> None:
    """Create the controlled feature-ablation result table."""
    if sa.inspect(op.get_bind()).has_table("robustness_experiments", schema=SCHEMA):
        return
    op.create_table(
        "robustness_experiments",
        sa.Column("experiment_run_id", sa.String(160), primary_key=True),
        sa.Column(
            "parent_run_id",
            sa.String(96),
            sa.ForeignKey(f"{SCHEMA}.training_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("experiment_group", sa.String(64), nullable=False),
        sa.Column("feature_set_version", sa.String(128), nullable=False),
        sa.Column("features_removed", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
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
        sa.Column("business_cost", sa.Numeric(20, 6), nullable=True),
        sa.Column("alert_rate", sa.Float(), nullable=False),
        sa.Column("top_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Drop geographic robustness experiment tracking."""
    if sa.inspect(op.get_bind()).has_table("robustness_experiments", schema=SCHEMA):
        op.drop_table("robustness_experiments", schema=SCHEMA)
