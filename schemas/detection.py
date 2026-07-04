from typing import Literal

from pydantic import BaseModel, Field

# NOTE: these must stay in sync with services/image/classifier.SKIN_CANCER_LABELS.
# We intentionally duplicate the values instead of importing from services/ to
# preserve the layering rule (schemas/ is Pydantic-only). A unit test asserts the
# two lists stay equal (tests/unit/test_detection.py).
SkinCancerLabel = Literal[
    "Melanoma",
    "Karsinoma Sel Basal",
    "Karsinoma Sel Skuamosa",
    "Nevus",
]


class DetectionResult(BaseModel):
    label: SkinCancerLabel = Field(..., description="EfficientNetB3 predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field("EfficientNetB3-v1", max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
