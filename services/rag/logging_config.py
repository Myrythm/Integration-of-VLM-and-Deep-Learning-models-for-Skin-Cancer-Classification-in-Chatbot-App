import hashlib
import json
import logging
import logging.handlers
from pathlib import Path

from config import PROJECT_ROOT

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"


def hash_query(query: str) -> str:
    """SHA256 hash of query for PII-safe logging."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def setup_logging(log_path: str | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = str(LOG_DIR / "rag_usage.log")
    formatter = logging.Formatter(LOG_FORMAT)

    # NOTE: TimedRotatingFileHandler is not multi-process safe — under
    # `uvicorn --workers N` each worker rotates at midnight and clobbers the others'
    # rotated files. Run single-worker, or ship logs to stdout and rotate externally.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=30, encoding="utf-8", delay=True
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    rag_logger = logging.getLogger("rag")
    rag_logger.handlers.clear()
    rag_logger.setLevel(logging.INFO)
    rag_logger.addHandler(file_handler)
    rag_logger.addHandler(console_handler)
    rag_logger.propagate = False


def log_query(
    session_id: str,
    query: str,
    language: str,
    num_chunks_retrieved: int,
    num_chunks_after_filter: int,
    citations_used: list[int],
    tokens_in: int,
    tokens_out: int,
    response_time_ms: int,
    classification: str = "answered",
) -> None:
    """Emit a structured log entry for a single query."""
    logger = logging.getLogger("rag")
    payload = {
        "event": "chat_query",
        "session_id": session_id,
        "query_hash": hash_query(query),
        "language": language,
        "classification": classification,
        "num_chunks_retrieved": num_chunks_retrieved,
        "num_chunks_after_filter": num_chunks_after_filter,
        "citations_used": citations_used,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "response_time_ms": response_time_ms,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))
