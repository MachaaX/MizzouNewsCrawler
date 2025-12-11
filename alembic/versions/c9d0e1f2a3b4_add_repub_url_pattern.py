"""add_repub_url_pattern

Revision ID: c9d0e1f2a3b4
Revises: 88f8f020d888
Create Date: 2025-12-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9d0e1f2a3b4'
down_revision: Union[str, Sequence[str], None] = '88f8f020d888'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Wire service patterns to add
# /repub/ URLs are republished syndicated content (e.g., missouriindependent.com/2025/12/10/repub/...)
# /eedition/ and /eeditions/ URLs are PDF replica pages, not articles
# /posterboard/ URLs are advertising sections
WIRE_PATTERNS = [
    {
        "service_name": "Republished Content",
        "pattern": "/repub/",
        "pattern_type": "url",
        "case_sensitive": False,
        "priority": 30,  # High priority - clear indicator of syndication
        "active": True,
        "notes": "URLs containing /repub/ are republished syndicated content from partner sites",
    },
    {
        "service_name": "E-Edition PDF",
        "pattern": "/eedition/",
        "pattern_type": "url",
        "case_sensitive": False,
        "priority": 20,  # Very high priority - not articles at all
        "active": True,
        "notes": "E-edition URLs are PDF replica newspaper pages, not article content",
    },
    {
        "service_name": "E-Edition PDF",
        "pattern": "/eeditions/",
        "pattern_type": "url",
        "case_sensitive": False,
        "priority": 20,  # Very high priority - not articles at all
        "active": True,
        "notes": "E-editions URLs are PDF replica newspaper pages, not article content",
    },
    {
        "service_name": "Advertising Content",
        "pattern": "/posterboard/",
        "pattern_type": "url",
        "case_sensitive": False,
        "priority": 20,  # Very high priority - advertising, not articles
        "active": True,
        "notes": "Posterboard URLs are advertising sections, not news articles",
    },
]


def upgrade() -> None:
    """Add /repub/ URL pattern for syndicated content detection."""
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
                    (service_name, pattern, pattern_type, case_sensitive, priority, active, notes)
                    VALUES (:service_name, :pattern, :pattern_type, :case_sensitive, :priority, :active, :notes)
                    """
                ),
                pattern,
            )


def downgrade() -> None:
    """Remove /repub/ URL pattern."""
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
