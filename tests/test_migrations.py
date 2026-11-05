import psycopg
from alembic import command
from conftest import alembic_config, alembic_upgrade

from contextread.models import Base


def tables(dsn):
    with psycopg.connect(dsn) as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        }


def test_migrations_match_models(test_dsn):
    assert tables(test_dsn) - {"alembic_version"} == set(Base.metadata.tables)


def test_downgrade_keeps_spend_ledger_and_upgrade_adopts_it(clean_db):
    test_dsn = clean_db  # starts from an empty ledger, whatever ran before
    with psycopg.connect(test_dsn) as conn:
        conn.execute("INSERT INTO llm_spend (model, status, cost_usd) VALUES ('m', 'done', 0.5)")
    command.downgrade(alembic_config(test_dsn), "base")
    assert tables(test_dsn) - {"alembic_version"} == {"llm_spend"}
    alembic_upgrade(test_dsn)
    assert tables(test_dsn) - {"alembic_version"} == set(Base.metadata.tables)
    with psycopg.connect(test_dsn) as conn:
        assert conn.execute("SELECT sum(cost_usd)::float FROM llm_spend").fetchone()[0] == 0.5
        conn.execute("TRUNCATE llm_spend")
