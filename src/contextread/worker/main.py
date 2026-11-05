"""Indexing worker: polls the job queue. One job at a time (the dev machine has 7 GB RAM)."""

import logging
import os
import signal
import socket
import time

from .. import (
    jobs,
    retrieval,  # noqa: F401  (registers the retrieval handler)
)
from ..indexing import pipeline  # noqa: F401  (registers the index_document handler)

log = logging.getLogger("contextread.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("worker %s started; handlers: %s", worker_id, sorted(jobs.HANDLERS))
    last_reclaim = 0.0
    while not stopping:
        if time.monotonic() - last_reclaim > 60:
            if n := jobs.reclaim_stale():
                log.warning("requeued %d stale jobs", n)
            last_reclaim = time.monotonic()
        if not jobs.run_one(worker_id):
            time.sleep(1.0)
    log.info("worker %s stopped", worker_id)


if __name__ == "__main__":
    main()
