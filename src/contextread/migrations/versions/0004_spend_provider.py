"""llm_spend provider: separate caps per LLM provider

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-02 09:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("llm_spend", sa.Column("provider", sa.Text(), server_default="openai", nullable=False))


def downgrade() -> None:
    op.drop_column("llm_spend", "provider")
