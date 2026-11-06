"""A document indexed without the worker (contextread index, a tool calling the pipeline) closes its upload's
queued job, so a worker started later does not index it again; a job a worker is running is left alone."""

from sqlalchemy import select, update
from test_indexing import DISNEY, upload

from contextread import jobs
from contextread.db import session_scope
from contextread.indexing import pipeline
from contextread.models import Job


def _jobs(engine, file_id):
    with session_scope(engine) as s:
        rows = s.scalars(select(Job).where(Job.payload["file_id"].astext == file_id)).all()
        return [(j.status, j.last_error) for j in rows]


def test_indexing_directly_closes_the_uploads_job(engine, clean_files, fake_llm, bucket):
    llm, _ = fake_llm
    reg = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    assert _jobs(engine, reg["doc_id"]) == [("queued", None)]
    pipeline.index_document(reg["doc_id"], llm=llm, engine=engine)
    assert _jobs(engine, reg["doc_id"]) == [("done", jobs.SUPERSEDED)]

    # a duplicate is copied from the first, and its job is closed the same way
    copy = upload(DISNEY, "copy.pdf", engine=engine)
    pipeline.index_document(copy["doc_id"], llm=llm, engine=engine)
    assert _jobs(engine, copy["doc_id"]) == [("done", jobs.SUPERSEDED)]


def test_a_running_job_is_left_alone(engine, clean_files, bucket):
    reg = upload(DISNEY, "q1-fy25-earnings.pdf", engine=engine)
    with session_scope(engine) as s:  # a worker has claimed it
        s.execute(update(Job).where(Job.payload["file_id"].astext == reg["doc_id"]).values(status="running"))
    assert jobs.close_queued("index_document", reg["doc_id"], engine=engine) == 0
    assert _jobs(engine, reg["doc_id"]) == [("running", None)]
