from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def mocked_validator():
    """Patch the VLM validator at the route module; default verdict 'valid'."""
    with patch("routes.api_routes.validate_skin_image", new_callable=AsyncMock) as mock_v:
        mock_v.return_value = "valid"
        yield mock_v


def test_upload_image_classifies_lesion(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detection" in body
    assert "chat_session_id" in body
    assert body["validation_status"] == "valid"
    mocked_classifier.assert_called_once()
    mocked_validator.assert_awaited_once()


def test_upload_rejects_non_lesion_image(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    mocked_validator.return_value = "invalid"
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 400
    assert "lesi kulit" in response.json()["detail"]
    mocked_classifier.assert_not_called()


def test_upload_uncertain_still_classifies(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    mocked_validator.return_value = "uncertain"
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["validation_status"] == "uncertain"
    mocked_classifier.assert_called_once()


def test_upload_fails_closed_when_validation_unavailable(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    from services.image.validator import ValidationUnavailableError

    mocked_validator.side_effect = ValidationUnavailableError("no key")
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 503
    mocked_classifier.assert_not_called()


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


def test_upload_rejects_truncated_image() -> None:
    full = _make_test_image_bytes(size=(200, 200))
    truncated = full[: len(full) // 2]
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(truncated), "image/png")},
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


def test_upload_image_hides_model_path_on_missing_model(mocked_validator: AsyncMock) -> None:
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


def test_upload_image_offloads_classification_to_thread(mocked_validator: AsyncMock) -> None:
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


def test_upload_image_hides_internal_error_details(mocked_validator: AsyncMock) -> None:
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
