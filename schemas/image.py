from typing import Literal

from pydantic import BaseModel

from schemas.detection import DetectionResult


class ImageUploadResponse(BaseModel):
    detection: DetectionResult
    chat_session_id: str
    validation_status: Literal["valid", "uncertain"] = "valid"
    message: str = "Image processed. Continue to chat for more info."
