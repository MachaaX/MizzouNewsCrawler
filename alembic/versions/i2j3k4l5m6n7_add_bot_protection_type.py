"""Add bot_protection_type and selenium_only columns to sources table.

Revision ID: i2j3k4l5m6n7
Revises: c9d0e1f2a3b4
Create Date: 2025-12-11

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "i2j3k4l5m6n7"
down_revision: Union[str, None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add bot_protection_type and selenium_only columns."""
    # Add bot_protection_type to identify specific bot protection
    # Values: 'perimeterx', 'cloudflare', 'datadome', 'akamai', etc.
    op.add_column(
        "sources",
        sa.Column("bot_protection_type", sa.String(64), nullable=True, index=True),
    )

    # Add selenium_only flag to force Selenium-only extraction
    # When True, skip HTTP requests and go directly to Selenium
    op.add_column(
        "sources",
        sa.Column(
            "selenium_only",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # Add bot_protection_detected_at to track when protection was first detected
    op.add_column(
        "sources",
        sa.Column("bot_protection_detected_at", sa.DateTime(), nullable=True),
    )

    # Create index on selenium_only for efficient filtering
    op.create_index(
        "ix_sources_selenium_only",
        "sources",
        ["selenium_only"],
        unique=False,
    )


def downgrade() -> None:
    """Remove bot_protection_type and selenium_only columns."""
    op.drop_index("ix_sources_selenium_only", table_name="sources")
    op.drop_column("sources", "bot_protection_detected_at")
    op.drop_column("sources", "selenium_only")
    op.drop_column("sources", "bot_protection_type")
