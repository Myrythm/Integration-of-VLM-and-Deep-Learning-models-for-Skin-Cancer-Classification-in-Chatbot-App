from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    label: str = Field(..., description="EfficientNetB3 predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = "EfficientNetB3-v1"
