import logging

from utils.rag.logging_config import hash_query, setup_logging


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
