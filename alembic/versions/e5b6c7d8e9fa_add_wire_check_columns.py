"""Add MediaCloud wire detection tracking columns.

Revision ID: e5b6c7d8e9fa
Revises: c3d4e5f6a7b8
Create Date: 2025-12-05 18:30:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e5b6c7d8e9fa"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None

PENDING_DEFAULT = "pending"
COMPLETE_DEFAULT = "complete"


def upgrade() -> None:
    op.add_column(
        "articles",
        sa.Column("wire_check_status", sa.String(), nullable=False, server_default=COMPLETE_DEFAULT),
    )
    op.add_column(
        "articles",
        sa.Column("wire_check_attempted_at", sa.DateTime(), nullable=True),
    )
    op.add_column(
        "articles",
        sa.Column("wire_check_error", sa.String(), nullable=True),
    )
    op.add_column(
        "articles",
        sa.Column("wire_check_metadata", sa.JSON(), nullable=True),
    )

    # Ensure existing rows are marked complete so legacy content is not reprocessed.
    op.execute("UPDATE articles SET wire_check_status = '{}'".format(COMPLETE_DEFAULT))

    # New rows should default to pending so they are eligible for MediaCloud processing.
    op.alter_column(
        "articles",
        "wire_check_status",
        server_default=PENDING_DEFAULT,
        existing_type=sa.String(),
    )

    # Composite index supports queue lookups by wire status + article status
    op.create_index(
        "ix_articles_wire_check_status",
        "articles",
        ["wire_check_status", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_articles_wire_check_status", table_name="articles")
    op.drop_column("articles", "wire_check_metadata")
    op.drop_column("articles", "wire_check_error")
    op.drop_column("articles", "wire_check_attempted_at")
    op.drop_column("articles", "wire_check_status")
