"""Tests run against a throwaway database (contextread_test) and bucket (contextread-test).
The canonical APP_PG_DSN is left untouched so live, billed calls still hit the real ledger."""

import os

import psycopg
import pytest
from alembic.config import Config
from fakes import FakeClient, completion
from sqlalchemy import create_engine

from contextread import config, db, migrate, storage
from contextread.db import session_scope
from contextread.llm import BudgetedLLM
from contextread.models import File, UsageEvent

os.environ["S3_BUCKET"] = "contextread-test"
# the reference free plan, whatever a local .env sets
os.environ["PAGE_LIMIT"], os.environ["CHAT_LIMIT"] = "1000", "100"
config.get_settings.cache_clear()
storage.get_client.cache_clear()


def _test_url() -> str:
    main = config.get_settings().database_url
    return main.rsplit("/", 1)[0] + "/contextread_test"


@pytest.fixture(scope="session")
def test_dsn() -> str:
    main = config.get_settings().database_url
    with psycopg.connect(main, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS contextread_test WITH (FORCE)")
        conn.execute("CREATE DATABASE contextread_test")
    dsn = _test_url()
    alembic_upgrade(dsn)
    return dsn


def alembic_config(dsn: str) -> Config:
    return migrate.alembic_config(dsn.replace("postgresql://", "postgresql+psycopg://", 1))


def alembic_upgrade(dsn: str) -> None:
    migrate.upgrade(dsn.replace("postgresql://", "postgresql+psycopg://", 1))


@pytest.fixture(scope="session")
def engine(test_dsn):
    eng = create_engine(test_dsn.replace("postgresql://", "postgresql+psycopg://", 1))
    db.override_engine(eng)  # code that uses the default engine (API, jobs, pipeline) hits the test DB
    yield eng
    db.override_engine(None)
    eng.dispose()


@pytest.fixture
def clean_db(test_dsn):
    with psycopg.connect(test_dsn) as conn:
        conn.execute("TRUNCATE jobs, llm_spend RESTART IDENTITY")
    return test_dsn


@pytest.fixture(scope="session")
def bucket():
    storage.ensure_bucket()
    return storage.bucket()


@pytest.fixture(scope="session", autouse=True)
def no_live_vision():
    """Indexing never calls a real vision model in tests (worker threads included); tests that
    need one pass a reader."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("contextread.indexing.pipeline.default_reader", lambda: None)
        yield


@pytest.fixture
def fake_llm(clean_db):
    client = FakeClient(
        resp=completion("A short factual summary."), headers={"x-litellm-response-cost": "0.00001"}
    )
    return BudgetedLLM(dsn=clean_db, cap_usd=1, client=client, cache=False), client


@pytest.fixture
def clean_files(test_dsn, engine):
    with session_scope(engine) as s:
        s.execute(File.__table__.delete())
        s.execute(UsageEvent.__table__.delete())
