"""Postgres-backed job queue: workers claim with FOR UPDATE SKIP LOCKED, so several
workers never take the same job and no extra broker is needed."""

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import Engine, func, select, update

from .db import get_engine, session_scope
from .models import Job

Handler = Callable[[dict[str, Any]], None]
HANDLERS: dict[str, Handler] = {}


def handler(kind: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return register


@handler("ping")
def _ping(payload: dict[str, Any]) -> None:
    if payload.get("fail"):
        raise RuntimeError("ping asked to fail")


def enqueue(
    kind: str, payload: dict[str, Any] | None = None, engine: Engine | None = None, max_attempts: int = 3
) -> int:
    with session_scope(engine) as s:
        job = Job(kind=kind, payload=payload or {}, max_attempts=max_attempts)
        s.add(job)
        s.flush()
        return job.id


def claim(worker_id: str, engine: Engine | None = None) -> Job | None:
    with session_scope(engine) as s:
        job = s.scalars(
            select(Job)
            .where(Job.status == "queued", Job.run_after <= func.now())
            .order_by(Job.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if job is None:
            return None
        job.status, job.locked_by, job.locked_at = "running", worker_id, func.now()
        job.attempts += 1
        s.flush()
        s.refresh(job)
        return job


def _finish(job_id: int, engine: Engine | None, **values: Any) -> None:
    with session_scope(engine) as s:
        s.execute(update(Job).where(Job.id == job_id).values(**values))


def run_one(worker_id: str, engine: Engine | None = None) -> bool:
    """Claim and run a single job. Returns False when the queue had nothing ready."""
    job = claim(worker_id, engine)
    if job is None:
        return False
    try:
        fn = HANDLERS.get(job.kind)
        if fn is None:
            raise LookupError(f"no handler for job kind {job.kind!r}")
        fn(job.payload)
    except Exception as e:
        retry = job.attempts < job.max_attempts
        _finish(
            job.id,
            engine,
            status="queued" if retry else "failed",
            last_error=f"{type(e).__name__}: {e}"[:2000],
            locked_by=None,
            locked_at=None,
            # exponential backoff: 2, 4, 8 … seconds
            run_after=func.now() + timedelta(seconds=2**job.attempts),
            finished_at=None if retry else func.now(),
        )
    else:
        _finish(job.id, engine, status="done", finished_at=func.now(), last_error=None)
    return True


def reclaim_stale(after: timedelta = timedelta(minutes=15), engine: Engine | None = None) -> int:
    """Requeue jobs whose worker died mid-run."""
    with session_scope(engine or get_engine()) as s:
        result = s.execute(
            update(Job)
            .where(Job.status == "running", Job.locked_at < func.now() - after)
            .values(status="queued", locked_by=None, locked_at=None)
        )
        return result.rowcount
