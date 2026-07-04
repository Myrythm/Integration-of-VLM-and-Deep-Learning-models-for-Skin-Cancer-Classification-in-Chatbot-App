import json
import logging
from unittest.mock import patch

from services.rag.logging_config import hash_query, log_query, setup_logging


def test_hash_query_deterministic() -> None:
    h1 = hash_query("apa itu melanoma?")
    h2 = hash_query("apa itu melanoma?")
    assert h1 == h2
    assert len(h1) == 64


def test_hash_query_obfuscates_pii() -> None:
    h1 = hash_query("halo nama saya John Doe, saya punya tahi lalat")
    h2 = hash_query("halo nama saya Jane Doe, saya punya tahi lalat")
    assert h1 != h2


def test_setup_logging_creates_logger() -> None:
    setup_logging(log_path="/tmp/test_rag.log")
    logger = logging.getLogger("rag")
    assert logger is not None
    assert logger.level <= logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) >= 1


def test_setup_logging_is_idempotent() -> None:
    """Repeated calls must not stack handlers (e.g. on worker reload or test rerun)."""
    setup_logging(log_path="/tmp/test_rag.log")
    setup_logging(log_path="/tmp/test_rag.log")
    setup_logging(log_path="/tmp/test_rag.log")
    logger = logging.getLogger("rag")
    assert len(logger.handlers) == 2


def test_log_query_records_classification() -> None:
    with patch("services.rag.logging_config.logging.getLogger") as mock_get:
        mock_logger = mock_get.return_value
        log_query(
            session_id="s",
            query="q",
            language="id",
            num_chunks_retrieved=3,
            num_chunks_after_filter=1,
            citations_used=[1],
            tokens_in=10,
            tokens_out=5,
            response_time_ms=1,
            classification="off_topic",
        )
    payload = json.loads(mock_logger.info.call_args[0][0])
    assert payload["classification"] == "off_topic"


def test_log_query_classification_defaults_to_answered() -> None:
    with patch("services.rag.logging_config.logging.getLogger") as mock_get:
        mock_logger = mock_get.return_value
        log_query(
            session_id="s",
            query="q",
            language="id",
            num_chunks_retrieved=0,
            num_chunks_after_filter=0,
            citations_used=[],
            tokens_in=0,
            tokens_out=0,
            response_time_ms=1,
        )
    payload = json.loads(mock_logger.info.call_args[0][0])
    assert payload["classification"] == "answered"
