from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app
from services.image.classifier import classify_skin_image

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


def test_upload_image_rejects_corrupt_image_bytes() -> None:
    with patch("services.image.classifier._get_model", return_value=MagicMock()):
        response = client.post(
            "/api/upload",
            files={"file": ("test.png", BytesIO(b"not a real image"), "image/png")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "File is not a valid image"


def test_upload_image_rejects_oversize_file() -> None:
    huge_bytes = b"0" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/upload",
        files={"file": ("big.png", BytesIO(huge_bytes), "image/png")},
    )
    assert response.status_code == 413


def test_upload_image_hides_model_path_on_missing_model() -> None:
    with patch(
        "services.image.classifier._get_model",
        side_effect=FileNotFoundError("/secret/path/model.h5"),
    ):
        response = client.post(
            "/api/upload",
            files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "Classification model unavailable"
    assert "/secret/path" not in detail


def test_upload_image_offloads_classification_to_thread() -> None:
    from schemas.detection import DetectionResult

    async def fake_to_thread(func, *args, **kwargs):
        assert func is classify_skin_image
        return DetectionResult(label="Melanoma", confidence=0.9)

    with patch("routes.api_routes.asyncio.to_thread", side_effect=fake_to_thread) as mock_to_thread:
        response = client.post(
            "/api/upload",
            files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
        )

    assert response.status_code == 200
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args[0]
    assert call_args[0] is classify_skin_image
    assert call_args[1] == _make_test_image_bytes()


def test_upload_image_hides_internal_error_details() -> None:
    with patch(
        "services.image.classifier._get_model",
        side_effect=RuntimeError("super secret internal detail"),
    ):
        response = client.post(
            "/api/upload",
            files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
        )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Internal classification error"
    assert "super secret internal detail" not in detail
