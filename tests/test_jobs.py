from sqlalchemy import select, update

from contextread import jobs
from contextread.db import session_scope
from contextread.models import Job


def get(engine, job_id) -> Job:
    with session_scope(engine) as s:
        return s.scalars(select(Job).where(Job.id == job_id)).one()


def test_job_runs_to_done(engine, clean_db):
    job_id = jobs.enqueue("ping", {"x": 1}, engine=engine)
    assert jobs.run_one("w1", engine) is True
    job = get(engine, job_id)
    assert (job.status, job.attempts, job.locked_by) == ("done", 1, "w1")
    assert jobs.run_one("w1", engine) is False


def test_failure_retries_with_backoff_then_fails(engine, clean_db):
    job_id = jobs.enqueue("ping", {"fail": True}, engine=engine, max_attempts=2)
    jobs.run_one("w1", engine)
    job = get(engine, job_id)
    assert job.status == "queued" and job.attempts == 1 and "ping asked to fail" in job.last_error
    assert jobs.run_one("w1", engine) is False  # backoff: not ready yet
    with session_scope(engine) as s:
        s.execute(update(Job).where(Job.id == job_id).values(run_after=Job.created_at))
    jobs.run_one("w1", engine)
    assert get(engine, job_id).status == "failed"


def test_unknown_kind_fails(engine, clean_db):
    job_id = jobs.enqueue("no-such-kind", engine=engine, max_attempts=1)
    jobs.run_one("w1", engine)
    job = get(engine, job_id)
    assert job.status == "failed" and "no handler" in job.last_error


def test_concurrent_claims_never_share_a_job(engine, clean_db):
    ids = {jobs.enqueue("ping", engine=engine) for _ in range(2)}
    a, b = jobs.claim("wa", engine), jobs.claim("wb", engine)
    assert {a.id, b.id} == ids
    assert jobs.claim("wc", engine) is None
