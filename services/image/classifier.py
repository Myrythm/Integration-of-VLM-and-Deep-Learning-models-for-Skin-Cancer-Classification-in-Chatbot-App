import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings, get_settings
from schemas.detection import DetectionResult

SKIN_CANCER_LABELS: list[str] = [
    "Karsinoma Sel Basal",
    "Karsinoma Sel Skuamosa",
    "Melanoma",
    "Nevus",
]

@lru_cache(maxsize=1)
def _get_model(model_path_str: str):
    from tensorflow.keras.models import load_model

    return load_model(model_path_str)


def preprocess_image(image_bytes: bytes, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def classify_skin_image(
    image_bytes: bytes,
    settings: Settings | None = None,
) -> DetectionResult:
    if settings is None:
        settings = get_settings()

    model = _get_model(settings.model_path)
    size = (settings.image_input_size, settings.image_input_size)
    img_array = preprocess_image(image_bytes, size=size)
    predictions = model.predict(img_array, verbose=0)

    if predictions.shape[-1] != len(SKIN_CANCER_LABELS):
        raise ValueError(
            f"Model output width {predictions.shape[-1]} does not match "
            f"{len(SKIN_CANCER_LABELS)} known labels"
        )

    probs = predictions[0]
    if not (0.99 <= float(probs.sum()) <= 1.01):
        probs = _softmax(probs)

    confidence = float(np.max(probs))
    predicted_idx = int(np.argmax(probs))
    label = SKIN_CANCER_LABELS[predicted_idx]

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
    )
