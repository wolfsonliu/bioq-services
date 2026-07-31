"""add users.role

Revision ID: f2c6bdf643d2
Revises: 7013f1926723
Create Date: 2026-07-31 16:29:47.631024
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2c6bdf643d2'
down_revision: str | None = '7013f1926723'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # server_default backfills existing rows (NOT NULL add on a populated table).
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('role', sa.String(), nullable=False, server_default='user')
        )


def downgrade() -> None:
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('role')
