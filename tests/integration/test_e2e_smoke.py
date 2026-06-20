import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_TESTS"),
    reason="Set RUN_LIVE_TESTS=1 to run live OpenAI + Chroma smoke test",
)
def test_e2e_chat_in_indonesian(tmp_path: Path) -> None:
    os.environ["CHROMA_PATH"] = str(tmp_path / "chroma_live")
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY_FOR_TEST", "missing")

    from main import app

    with TestClient(app) as client:
        from utils.rag.app_state import initialize_app_state
        initialize_app_state()

        from utils.rag.ingestion import ingest_directory
        ingest_directory("aad")

        with client.stream(
            "POST",
            "/api/chat",
            json={
                "session_id": "live-test-1",
                "message": "Apa itu melanoma?",
                "detection": {"label": "benign_nevus", "confidence": 0.85, "model_version": "v1"},
            },
        ) as response:
            assert response.status_code == 200
            tokens = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    import json
                    payload = json.loads(line[6:])
                    if payload.get("type") == "token":
                        tokens.append(payload.get("content", ""))
            full = "".join(tokens)
            assert "melanoma" in full.lower()
            assert "⚠️" in full or "edukasi" in full.lower() or "consult" in full.lower()
