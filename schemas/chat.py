from pydantic import BaseModel, Field

from schemas.detection import DetectionResult


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    detection: DetectionResult | None = None


class ChatChunk(BaseModel):
    type: str
    content: str | None = None
    citations: list[dict] | None = None
    language: str | None = None


class CitationOut(BaseModel):
    number: int
    title: str
    url: str
    source: str
