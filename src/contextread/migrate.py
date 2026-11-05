"""Database migrations shipped inside the package: `contextread migrate` upgrades APP_PG_DSN to the
latest schema, and tests upgrade their throwaway database the same way."""

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    """Alembic settings without an alembic.ini; `url` (a SQLAlchemy URL) overrides APP_PG_DSN."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("file_template", "%%(rev)s_%%(slug)s")
    if url:
        cfg.attributes["url"] = url
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)
