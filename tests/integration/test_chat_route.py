import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

_SESSION_ID = "11111111-1111-4111-8111-111111111111"


def _token_event(content: str, usage: dict | None = None) -> dict:
    chunk = MagicMock()
    chunk.content = content
    chunk.usage_metadata = usage
    return {
        "event": "on_chat_model_stream",
        "name": "ChatOpenAI",
        "run_id": "llm",
        "tags": [],
        "metadata": {},
        "data": {"chunk": chunk},
    }


def _retriever_end_event(retrieved_docs: list[Document], name: str = "retrieve") -> dict:
    return {
        "event": "on_chain_end",
        "name": name,
        "run_id": "retrieve",
        "tags": [],
        "metadata": {},
        "data": {
            "output": {
                "question": "apa itu melanoma?",
                "chat_history": [],
                "detection": "...",
                "language": "id",
                "disclaimer": "...",
                "context": "...",
                "retrieved_docs": retrieved_docs,
            }
        },
    }


def _build_chain_mock(
    retrieved_docs: list[Document],
    tokens: list[str],
    retrieve_name: str = "retrieve",
    usage: dict | None = None,
    captured: dict | None = None,
) -> MagicMock:
    mock_chain = MagicMock()

    async def fake_astream_events(_input, _config=None, version=None):
        if captured is not None:
            captured["input"] = _input
        yield _retriever_end_event(retrieved_docs, name=retrieve_name)
        for t in tokens:
            yield _token_event(t)
        if usage is not None:
            yield _token_event("", usage=usage)

    mock_chain.astream_events = fake_astream_events
    return mock_chain


def _make_patches(mock_chain: MagicMock):
    mock_memory = MagicMock()
    mock_memory.get_history = AsyncMock(return_value=[])
    mock_memory.add_turn = AsyncMock()
    mock_settings = MagicMock()
    mock_settings.rag_similarity_threshold = 0.7
    mock_settings.rag_top_k = 5
    mock_settings.rag_retrieve_k = 10
    return (
        patch("services.rag.app_state._chain", mock_chain),
        patch("services.rag.app_state._memory", mock_memory),
        patch("services.rag.app_state._settings", mock_settings),
    ), mock_memory


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def _post(client: TestClient, **overrides) -> object:
    payload = {
        "session_id": _SESSION_ID,
        "message": "apa itu melanoma?",
        "detection": {"label": "Nevus", "confidence": 0.85, "model_version": "v1"},
    }
    payload.update(overrides)
    return client.post("/api/chat", json=payload)


def test_chat_endpoint_returns_sse_stream() -> None:
    mock_chain = _build_chain_mock(retrieved_docs=[], tokens=["Halo ", "dokter"])
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        response = _post(TestClient(app))
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _parse_sse(response.text)
        assert events[-1]["type"] == "done"
    finally:
        p1.stop(); p2.stop(); p3.stop()


def test_sse_response_carries_antibuffering_headers() -> None:
    mock_chain = _build_chain_mock(retrieved_docs=[], tokens=["hi"])
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        response = _post(TestClient(app))
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
    finally:
        p1.stop(); p2.stop(); p3.stop()


def test_citation_emitted_after_tokens_filtered_to_cited() -> None:
    docs = [
        Document(page_content="Melanoma serius.", metadata={"title": "Melanoma Overview", "url": "https://x/m", "source": "aad"}),
        Document(page_content="BCC.", metadata={"title": "BCC", "url": "https://x/b", "source": "medlineplus"}),
        Document(page_content="SCC.", metadata={"title": "SCC", "url": "https://x/s", "source": "dermnet"}),
    ]
    mock_chain = _build_chain_mock(retrieved_docs=docs, tokens=["Jawaban ", "[1]", " selesai"])
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        response = _post(TestClient(app))
        events = _parse_sse(response.text)
        types = [e["type"] for e in events]
        citation_events = [e for e in events if e["type"] == "citation"]
        assert len(citation_events) == 1
        cits = citation_events[0]["citations"]
        assert len(cits) == 1
        assert cits[0]["number"] == 1
        assert cits[0]["title"] == "Melanoma Overview"
        # citation comes after the last token and before done
        assert types.index("citation") > max(i for i, t in enumerate(types) if t == "token")
        assert types.index("citation") < types.index("done")
    finally:
        p1.stop(); p2.stop(); p3.stop()


def test_on_chain_end_with_other_name_is_ignored() -> None:
    docs = [Document(page_content="x", metadata={"title": "T", "url": "", "source": "aad"})]
    # retriever event carries a different run name -> route must not capture its docs
    mock_chain = _build_chain_mock(retrieved_docs=docs, tokens=["a ", "[1]"], retrieve_name="SomethingElse")
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        response = _post(TestClient(app))
        events = _parse_sse(response.text)
        assert [e for e in events if e["type"] == "citation"] == []
    finally:
        p1.stop(); p2.stop(); p3.stop()


def test_error_path_hides_internals_and_emits_done() -> None:
    mock_chain = MagicMock()

    async def boom(_input, _config=None, version=None):
        raise RuntimeError("super secret internal detail")
        yield  # pragma: no cover - makes this an async generator

    mock_chain.astream_events = boom
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        response = _post(TestClient(app))
        assert response.status_code == 200
        assert "super secret internal detail" not in response.text
        events = _parse_sse(response.text)
        types = [e["type"] for e in events]
        assert "error" in types
        assert types[-1] == "done"
    finally:
        p1.stop(); p2.stop(); p3.stop()


def test_timeout_emits_error_and_done() -> None:
    import asyncio

    mock_chain = MagicMock()

    async def slow(_input, _config=None, version=None):
        await asyncio.sleep(0.5)
        yield _token_event("late")

    mock_chain.astream_events = slow
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p4 = patch(
        "routes.chat_routes.get_settings",
        return_value=SimpleNamespace(chat_stream_timeout_seconds=0.01),
    )
    p1.start(); p2.start(); p3.start(); p4.start()
    try:
        from main import app
        response = _post(TestClient(app))
        events = _parse_sse(response.text)
        types = [e["type"] for e in events]
        assert "error" in types
        assert types[-1] == "done"
    finally:
        p1.stop(); p2.stop(); p3.stop(); p4.stop()


def test_usage_metadata_recorded_in_log() -> None:
    mock_chain = _build_chain_mock(
        retrieved_docs=[],
        tokens=["Halo"],
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    (p1, p2, p3), _ = _make_patches(mock_chain)
    with patch("routes.chat_routes.log_query") as mock_log:
        p1.start(); p2.start(); p3.start()
        try:
            from main import app
            _post(TestClient(app))
        finally:
            p1.stop(); p2.stop(); p3.stop()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["tokens_in"] == 10
    assert kwargs["tokens_out"] == 5


def test_log_query_reports_distinct_prefilter_count() -> None:
    doc = Document(
        page_content="Melanoma.",
        metadata={"title": "M", "url": "", "source": "aad", "prefilter_count": 3},
    )
    mock_chain = _build_chain_mock(retrieved_docs=[doc], tokens=["a ", "[1]"])
    (p1, p2, p3), _ = _make_patches(mock_chain)
    with patch("routes.chat_routes.log_query") as mock_log:
        p1.start(); p2.start(); p3.start()
        try:
            from main import app
            _post(TestClient(app))
        finally:
            p1.stop(); p2.stop(); p3.stop()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["num_chunks_retrieved"] == 3
    assert kwargs["num_chunks_after_filter"] == 1
    assert kwargs["citations_used"] == [1]


def test_detection_rendered_safely_into_chain_input() -> None:
    captured: dict = {}
    mock_chain = _build_chain_mock(retrieved_docs=[], tokens=["ok"], captured=captured)
    (p1, p2, p3), _ = _make_patches(mock_chain)
    p1.start(); p2.start(); p3.start()
    try:
        from main import app
        _post(TestClient(app))
    finally:
        p1.stop(); p2.stop(); p3.stop()
    assert captured["input"]["detection"] == "label=Nevus, confidence=0.8500"


def test_blocked_query_appends_disclaimer_and_logs() -> None:
    from services.rag.disclaimer import DISCLAIMERS
    from services.rag.language import detect_language

    message = "berapa mg dosis obat yang harus saya minum setiap hari?"
    expected_lang = detect_language(message)

    mock_chain = _build_chain_mock(retrieved_docs=[], tokens=[])
    (p1, p2, p3), mock_memory = _make_patches(mock_chain)
    with patch("routes.chat_routes.log_query") as mock_log:
        p1.start(); p2.start(); p3.start()
        try:
            from main import app
            response = _post(TestClient(app), message=message)
            events = _parse_sse(response.text)
        finally:
            p1.stop(); p2.stop(); p3.stop()
    blocked = [e for e in events if e["type"] == "blocked"]
    assert len(blocked) == 1
    assert blocked[0]["content"].endswith(DISCLAIMERS[expected_lang])
    assert events[-1]["type"] == "done"
    mock_memory.add_turn.assert_awaited_once()
    assert mock_log.call_args.kwargs["classification"] == "unsafe_dosage"
