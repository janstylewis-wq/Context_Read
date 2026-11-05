from alembic import context
from sqlalchemy import create_engine

from contextread.config import get_settings
from contextread.models import Base

target_metadata = Base.metadata


def _url() -> str:
    # tests and autogenerate pass an explicit URL; everything else migrates APP_PG_DSN
    return context.config.attributes.get("url") or get_settings().sqlalchemy_url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
