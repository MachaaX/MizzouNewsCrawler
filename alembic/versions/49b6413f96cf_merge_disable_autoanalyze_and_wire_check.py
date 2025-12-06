"""Merge disable autoanalyze and wire check

Revision ID: 49b6413f96cf
Revises: 995983225474, e5b6c7d8e9fa
Create Date: 2025-12-05 18:09:01.099323

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '49b6413f96cf'
down_revision: Union[str, Sequence[str], None] = ('995983225474', 'e5b6c7d8e9fa')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
