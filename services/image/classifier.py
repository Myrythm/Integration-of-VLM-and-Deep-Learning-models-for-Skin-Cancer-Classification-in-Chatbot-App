import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from config import Settings, get_settings
from schemas.detection import DetectionResult

SKIN_CANCER_LABELS: list[str] = [
    "Melanoma",
    "Karsinoma Sel Basal",
    "Karsinoma Sel Skuamosa",
    "Nevus",
]

INPUT_SIZE: tuple[int, int] = (224, 224)


@lru_cache(maxsize=1)
def _get_model(model_path_str: str):
    from tensorflow.keras.models import load_model

    return load_model(model_path_str)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_skin_image(
    image_bytes: bytes,
    settings: Settings | None = None,
) -> DetectionResult:
    if settings is None:
        settings = get_settings()

    model = _get_model(settings.model_path)
    img_array = preprocess_image(image_bytes)
    predictions = model.predict(img_array, verbose=0)
    confidence = float(np.max(predictions))
    predicted_idx = int(np.argmax(predictions))
    label = SKIN_CANCER_LABELS[predicted_idx]

    return DetectionResult(
        label=label,
        confidence=round(confidence, 4),
    )
