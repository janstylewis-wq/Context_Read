from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings

_override: Engine | None = None


def override_engine(engine: Engine | None) -> None:
    """Point every default-engine caller (API, jobs, pipeline) at another database (tests)."""
    global _override
    _override = engine


def get_engine(url: str | None = None) -> Engine:
    if url is None and _override is not None:
        return _override
    return _engine(url or get_settings().sqlalchemy_url)


@lru_cache
def _engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """A session that commits on success and rolls back on error."""
    session = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
