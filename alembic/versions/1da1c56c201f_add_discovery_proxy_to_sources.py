"""add discovery_proxy to sources

Revision ID: 1da1c56c201f
Revises: a323ff14aed4
Create Date: 2025-12-20 16:35:13.203579

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1da1c56c201f'
down_revision: Union[str, Sequence[str], None] = 'a323ff14aed4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add discovery_proxy column to sources table
    op.add_column(
        'sources',
        sa.Column('discovery_proxy', sa.String(length=32), nullable=True)
    )
    # Add index for performance
    op.create_index(
        'ix_sources_discovery_proxy',
        'sources',
        ['discovery_proxy'],
        unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove index
    op.drop_index('ix_sources_discovery_proxy', table_name='sources')
    # Remove column
    op.drop_column('sources', 'discovery_proxy')
