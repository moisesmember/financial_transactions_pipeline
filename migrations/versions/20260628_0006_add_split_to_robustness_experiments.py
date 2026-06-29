"""Track robustness experiment metrics per temporal split.

Revision ID: 20260628_0006
Revises: 20260614_0005
Create Date: 2026-06-28
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260628_0006"
down_revision: str | None = "20260614_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "fraud_tracking"


def upgrade() -> None:
    """Change robustness experiments from one row per group to one row per split."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("robustness_experiments", schema=SCHEMA):
        return
    columns = {column["name"] for column in inspector.get_columns("robustness_experiments", schema=SCHEMA)}
    if "split" not in columns:
        op.add_column(
            "robustness_experiments",
            sa.Column("split", sa.String(24), nullable=False, server_default="unknown"),
            schema=SCHEMA,
        )
    op.drop_constraint(
        "robustness_experiments_pkey",
        "robustness_experiments",
        schema=SCHEMA,
        type_="primary",
    )
    op.create_primary_key(
        "pk_robustness_experiments",
        "robustness_experiments",
        ["experiment_run_id", "split"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    """Restore the previous single-row-per-experiment key."""
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("robustness_experiments", schema=SCHEMA):
        return
    op.drop_constraint(
        "pk_robustness_experiments",
        "robustness_experiments",
        schema=SCHEMA,
        type_="primary",
    )
    op.create_primary_key(
        "robustness_experiments_pkey",
        "robustness_experiments",
        ["experiment_run_id"],
        schema=SCHEMA,
    )
    columns = {column["name"] for column in inspector.get_columns("robustness_experiments", schema=SCHEMA)}
    if "split" in columns:
        op.drop_column("robustness_experiments", "split", schema=SCHEMA)
