import pytest
from pydantic import ValidationError

from schemas.chat import ChatChunk, ChatRequest, CitationOut


def test_chatchunk_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        ChatChunk(type="bogus")


def test_chatchunk_accepts_known_types() -> None:
    for t in ("token", "citation", "blocked", "done", "error"):
        assert ChatChunk(type=t).type == t


def test_chatchunk_coerces_citation_dicts_to_citationout() -> None:
    chunk = ChatChunk(
        type="citation",
        citations=[{"number": 1, "title": "t", "url": "", "source": "aad"}],
    )
    assert isinstance(chunk.citations[0], CitationOut)
    assert chunk.citations[0].number == 1


def test_chatrequest_rejects_malformed_session_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(session_id="abc", message="hi")


def test_chatrequest_accepts_uuid_session_id() -> None:
    req = ChatRequest(session_id="11111111-1111-4111-8111-111111111111", message="hi")
    assert req.session_id == "11111111-1111-4111-8111-111111111111"
