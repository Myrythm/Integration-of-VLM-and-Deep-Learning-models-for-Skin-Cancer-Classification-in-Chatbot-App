from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

client = TestClient(app)


def _make_test_image_bytes(size: tuple[int, int] = (10, 10), color: str = "red") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mocked_classifier():
    """Patch the classifier at the route module so the integration test does
    not need TensorFlow installed at test time."""
    from schemas.detection import DetectionResult

    with patch("routes.api_routes.classify_skin_image") as mock_classify:
        mock_classify.return_value = DetectionResult(
            label="Melanoma",
            confidence=0.87,
        )
        yield mock_classify


def test_upload_image_classifies_lesion(mocked_classifier: MagicMock) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detection" in body
    assert "chat_session_id" in body
    mocked_classifier.assert_called_once()


def test_upload_image_rejects_unsupported_type(mocked_classifier: MagicMock) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("test.bmp", BytesIO(_make_test_image_bytes()), "image/bmp")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    mocked_classifier.assert_not_called()
