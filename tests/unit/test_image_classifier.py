from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from config import Settings
from services.image.classifier import (
    SKIN_CANCER_LABELS,
    classify_skin_image,
    preprocess_image,
)


def _make_test_image_bytes(size: tuple[int, int] = (500, 700), color: str = "red") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_labels_are_four_classes() -> None:
    assert len(SKIN_CANCER_LABELS) == 4
    assert "Melanoma" in SKIN_CANCER_LABELS
    assert "Nevus" in SKIN_CANCER_LABELS


def test_preprocess_image_returns_correct_shape() -> None:
    arr = preprocess_image(_make_test_image_bytes())

    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.max() <= 255.0 and arr.min() >= 0.0


def test_preprocess_image_converts_rgba_to_rgb() -> None:
    rgba_buf = BytesIO()
    Image.new("RGBA", (300, 300), color=(255, 0, 0, 128)).save(rgba_buf, format="PNG")
    arr = preprocess_image(rgba_buf.getvalue())
    assert arr.shape == (1, 224, 224, 3)


def test_classify_skin_image_uses_model_prediction() -> None:
    settings = Settings(openai_api_key="test", model_path="./model/skinCancer.h5")
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.05, 0.10, 0.80, 0.05]])

    with patch("services.image.classifier._get_model", return_value=fake_model):
        result = classify_skin_image(_make_test_image_bytes(), settings=settings)

    assert result.label == "Melanoma"
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == pytest.approx(0.80, abs=1e-6)
    fake_model.predict.assert_called_once()
    call_arg = fake_model.predict.call_args[0][0]
    assert call_arg.shape == (1, 224, 224, 3)


def test_classify_skin_image_picks_max_class() -> None:
    settings = Settings(openai_api_key="test", model_path="./model/skinCancer.h5")
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.0, 0.0, 0.0, 1.0]])

    with patch("services.image.classifier._get_model", return_value=fake_model):
        result = classify_skin_image(_make_test_image_bytes(), settings=settings)

    assert result.label == "Nevus"
    assert result.confidence == pytest.approx(1.0, abs=1e-6)


def test_classify_skin_image_handles_low_confidence() -> None:
    settings = Settings(openai_api_key="test", model_path="./model/skinCancer.h5")
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.25, 0.25, 0.25, 0.25]])

    with patch("services.image.classifier._get_model", return_value=fake_model):
        result = classify_skin_image(_make_test_image_bytes(), settings=settings)

    assert result.confidence == pytest.approx(0.25, abs=1e-6)
    assert result.label in SKIN_CANCER_LABELS
