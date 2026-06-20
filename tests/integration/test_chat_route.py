import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("utils.rag.app_state._chain") as mock_chain, \
         patch("utils.rag.app_state._memory") as mock_memory, \
         patch("utils.rag.app_state._settings") as mock_settings:
        mock_settings.rag_similarity_threshold = 0.7
        mock_settings.rag_top_k = 5
        mock_settings.rag_retrieve_k = 10

        async def fake_astream(_inputs):
            yield MagicMock(content="Halo ")
            yield MagicMock(content="dokter")

        mock_chain.astream = fake_astream
        mock_memory.get_history.return_value = []

        from main import app
        yield TestClient(app)


def test_chat_endpoint_returns_sse_stream(client: TestClient) -> None:
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
