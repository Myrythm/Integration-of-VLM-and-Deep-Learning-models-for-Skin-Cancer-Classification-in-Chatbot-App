from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_when_chain_not_initialized():
    with patch("services.rag.app_state._chain", None):
        response = TestClient(app).get("/ready")
    assert response.status_code == 503


def test_ready_returns_200_when_chain_initialized():
    with patch("services.rag.app_state._chain", MagicMock()):
        response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_initialize_app_state_requires_openai_key():
    import services.rag.app_state as app_state
    from config import Settings

    with patch(
        "services.rag.app_state.get_settings",
        return_value=Settings(openai_api_key="", llm_backend="openai"),
    ):
        with pytest.raises(ValueError):
            app_state.initialize_app_state()
