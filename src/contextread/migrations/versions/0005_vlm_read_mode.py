"""pages.read_mode 'vlm': pages read by a vision model

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-02 10:10:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("pages_read_mode_check", "pages", type_="check")
    op.create_check_constraint("pages_read_mode_check", "pages", "read_mode IN ('text', 'image', 'vlm')")


def downgrade() -> None:
    op.execute("UPDATE pages SET read_mode = 'image' WHERE read_mode = 'vlm'")
    op.drop_constraint("pages_read_mode_check", "pages", type_="check")
    op.create_check_constraint("pages_read_mode_check", "pages", "read_mode IN ('text', 'image')")
