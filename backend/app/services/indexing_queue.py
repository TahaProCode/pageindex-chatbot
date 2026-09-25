"""
Multi-worker indexing queue.

Documents are processed with controlled concurrency (default 3 workers)
so a 200-file batch finishes in a fraction of the time of a single-threaded
queue, while still protecting the local Ollama instance from being
saturated by dozens of concurrent indexing jobs.
"""
import logging
import queue
import threading

from app.core.config import settings
from app.services.indexing_service import process_document

logger = logging.getLogger("pageindex.queue")

_queue: "queue.Queue[str]" = queue.Queue()
_workers_started = False
_lock = threading.Lock()

# Safe default if config.py was not updated yet
DEFAULT_MAX_CONCURRENT_INDEXING = 3


def _worker_loop(worker_id: int) -> None:
    while True:
        document_id = _queue.get()
        try:
            logger.info(
                "Indexing worker-%d: starting document %s (%d remaining in queue)",
                worker_id, document_id, _queue.qsize(),
            )
            process_document(document_id)
            logger.info(
                "Indexing worker-%d: finished document %s",
                worker_id, document_id,
            )
        except Exception:
            logger.exception(
                "Indexing worker-%d: unhandled error processing document %s",
                worker_id, document_id,
            )
        finally:
            _queue.task_done()


def start_worker() -> None:
    """
    Call once at app startup. Spawns concurrent worker threads.
    Safe to call multiple times (idempotent).
    """
    global _workers_started
    with _lock:
        if _workers_started:
            return

        n_workers = max(
            1,
            getattr(settings, "max_concurrent_indexing", DEFAULT_MAX_CONCURRENT_INDEXING),
        )
        for i in range(n_workers):
            t = threading.Thread(
                target=_worker_loop,
                args=(i + 1,),
                daemon=True,
                name=f"indexing-worker-{i + 1}",
            )
            t.start()

        _workers_started = True
        logger.info(
            "Indexing workers started: %d concurrent workers",
            n_workers,
        )


def enqueue_document(document_id: str) -> int:
    """Adds a document to the indexing queue. Returns the new queue size."""
    _queue.put(document_id)
    return _queue.qsize()


def queue_size() -> int:
    """Current number of documents waiting (not including those in-flight)."""
    return _queue.qsize()