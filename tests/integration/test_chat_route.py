import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document


def _token_event(content: str) -> dict:
    chunk = MagicMock()
    chunk.content = content
    return {
        "event": "on_llm_stream",
        "name": "ChatOpenAI",
        "run_id": "llm",
        "tags": [],
        "metadata": {},
        "data": {"chunk": chunk, "chunk_type": "AIMessageChunk"},
    }


def _retriever_end_event(retrieved_docs: list[Document]) -> dict:
    return {
        "event": "on_chain_end",
        "name": "RunnableLambda",
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


def _llm_end_event() -> dict:
    final = MagicMock()
    final.content = "Halo [1] dokter"
    return {
        "event": "on_llm_end",
        "name": "ChatOpenAI",
        "run_id": "llm",
        "tags": [],
        "metadata": {},
        "data": {"output": final},
    }


def _build_chain_mock(retrieved_docs: list[Document], tokens: list[str]) -> MagicMock:
    mock_chain = MagicMock()

    async def fake_astream_events(_input, _config=None, version=None):
        yield _retriever_end_event(retrieved_docs)
        for t in tokens:
            yield _token_event(t)
        yield _llm_end_event()

    mock_chain.astream_events = fake_astream_events
    return mock_chain


def _make_patches(mock_chain: MagicMock):
    mock_memory = MagicMock()
    mock_memory.get_history.return_value = []
    mock_settings = MagicMock()
    mock_settings.rag_similarity_threshold = 0.7
    mock_settings.rag_top_k = 5
    mock_settings.rag_retrieve_k = 10
    return (
        patch("utils.rag.app_state._chain", mock_chain),
        patch("utils.rag.app_state._memory", mock_memory),
        patch("utils.rag.app_state._settings", mock_settings),
    )


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_chat_endpoint_returns_sse_stream() -> None:
    mock_chain = _build_chain_mock(retrieved_docs=[], tokens=["Halo ", "dokter"])
    p1, p2, p3 = _make_patches(mock_chain)
    p1.start()
    p2.start()
    p3.start()
    try:
        from main import app
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "session_id": "test-session-1",
                "message": "apa itu melanoma?",
                "detection": {"label": "benign_nevus", "confidence": 0.85, "model_version": "v1"},
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
    finally:
        p1.stop()
        p2.stop()
        p3.stop()


def test_chat_endpoint_emits_citation_chunk_when_retriever_returns_docs() -> None:
    docs = [
        Document(
            page_content="Melanoma adalah kanker kulit yang serius.",
            metadata={
                "title": "Melanoma Overview",
                "url": "https://example.com/melanoma",
                "source": "aad",
            },
        ),
    ]
    mock_chain = _build_chain_mock(
        retrieved_docs=docs,
        tokens=["Halo ", "[1] ", "dokter"],
    )
    p1, p2, p3 = _make_patches(mock_chain)
    p1.start()
    p2.start()
    p3.start()
    try:
        from main import app
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "session_id": "test-session-1",
                "message": "apa itu melanoma?",
                "detection": {"label": "benign_nevus", "confidence": 0.85, "model_version": "v1"},
            },
        )
        assert response.status_code == 200
        events = _parse_sse(response.text)
        citation_events = [e for e in events if e.get("type") == "citation"]
        assert len(citation_events) >= 1
        citations = citation_events[0]["citations"]
        assert any(c["title"] == "Melanoma Overview" for c in citations)
        assert any(c["source"] == "aad" for c in citations)
        assert any(c["url"] == "https://example.com/melanoma" for c in citations)
    finally:
        p1.stop()
        p2.stop()
        p3.stop()
