"""add_talker_news_and_daypop_wire_patterns

Revision ID: 88f8f020d888
Revises: 49b6413f96cf
Create Date: 2025-12-09 11:08:12.044874

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '88f8f020d888'
down_revision: Union[str, Sequence[str], None] = '49b6413f96cf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Wire service patterns to add for Talker News and Daypop syndicators
WIRE_PATTERNS = [
    # Talker News - science/health content syndicator via BLOX Digital Content Exchange
    {
        "service_name": "Talker News",
        "pattern": r"\bTalker News\b",
        "pattern_type": "author",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
    {
        "service_name": "Talker News",
        "pattern": r"^Talker$",
        "pattern_type": "author",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
    {
        "service_name": "Talker News",
        "pattern": r"talker\.news",
        "pattern_type": "content",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
    {
        "service_name": "Talker News",
        "pattern": r"Originally published on talker\.news",
        "pattern_type": "content",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
    {
        "service_name": "Talker News",
        "pattern": "talker.news",
        "pattern_type": "url",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
    # Daypop - entertainment/news syndicator used by InterTech Media radio stations
    # Appears in JSON-LD articleSection field (e.g., "Entertainment Daypop", "News Daypop")
    {
        "service_name": "Daypop",
        "pattern": r"Daypop",
        "pattern_type": "content",
        "case_sensitive": False,
        "priority": 50,
        "active": True,
    },
]


def upgrade() -> None:
    """Add Talker News and Daypop wire service patterns."""
    conn = op.get_bind()

    for pattern in WIRE_PATTERNS:
        # Check if pattern already exists (idempotent)
        result = conn.execute(
            sa.text(
                """
                SELECT id FROM wire_services
                WHERE service_name = :service_name
                  AND pattern = :pattern
                  AND pattern_type = :pattern_type
                """
            ),
            {
                "service_name": pattern["service_name"],
                "pattern": pattern["pattern"],
                "pattern_type": pattern["pattern_type"],
            },
        ).fetchone()

        if result is None:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO wire_services
                    (service_name, pattern, pattern_type, case_sensitive, priority, active)
                    VALUES (:service_name, :pattern, :pattern_type, :case_sensitive, :priority, :active)
                    """
                ),
                pattern,
            )


def downgrade() -> None:
    """Remove Talker News and Daypop wire service patterns."""
    conn = op.get_bind()

    for pattern in WIRE_PATTERNS:
        conn.execute(
            sa.text(
                """
                DELETE FROM wire_services
                WHERE service_name = :service_name
                  AND pattern = :pattern
                  AND pattern_type = :pattern_type
                """
            ),
            {
                "service_name": pattern["service_name"],
                "pattern": pattern["pattern"],
                "pattern_type": pattern["pattern_type"],
            },
        )
