### Indexing_queue.py

"""
Ensures documents are indexed strictly one at a time, no matter how many
uploads arrive close together (e.g. a 200-file batch upload). A single
dedicated worker thread pulls document ids off a FIFO queue and processes
them sequentially - this keeps Ollama from being hit with dozens of
concurrent indexing requests at once.
"""
import logging
import queue
import threading

from app.services.indexing_service import process_document

logger = logging.getLogger("pageindex.queue")

_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False
_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        document_id = _queue.get()
        try:
            logger.info(
                "Indexing queue: starting document %s (%d remaining after this)",
                document_id, _queue.qsize(),
            )
            process_document(document_id)
        except Exception:
            logger.exception("Indexing queue: unhandled error processing document %s", document_id)
        finally:
            _queue.task_done()


def start_worker() -> None:
    """Call once at app startup. Safe to call multiple times."""
    global _worker_started
    with _lock:
        if not _worker_started:
            t = threading.Thread(target=_worker_loop, daemon=True, name="indexing-worker")
            t.start()
            _worker_started = True
            logger.info("Indexing worker thread started")


def enqueue_document(document_id: str) -> int:
    """Adds a document to the indexing queue. Returns the new queue size."""
    _queue.put(document_id)
    return _queue.qsize()