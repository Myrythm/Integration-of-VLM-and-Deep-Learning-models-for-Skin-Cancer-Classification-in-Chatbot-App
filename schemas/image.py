from pydantic import BaseModel

from schemas.detection import DetectionResult


class ImageUploadResponse(BaseModel):
    detection: DetectionResult
    chat_session_id: str
    message: str = "Image processed. Continue to chat for more info."
