import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from schemas.detection import DetectionResult


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)
    detection: DetectionResult | None = None

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        # Sessions are minted server-side as UUID4 by /api/upload; reject anything
        # that is not a well-formed UUID so histories can't be poisoned by guessing
        # ids and arbitrarily long ids can't become unbounded dict keys.
        uuid.UUID(value)
        return value


class CitationOut(BaseModel):
    number: int
    title: str
    url: str
    source: str


class ChatChunk(BaseModel):
    """A single Server-Sent Event frame for POST /api/chat.

    Event types (`type`) and the fields that accompany each:
      - ``token``    : incremental answer text in ``content``.
      - ``citation`` : ``citations`` lists the sources the answer actually cited.
      - ``blocked``  : ``content`` holds a canned refusal (safety block).
      - ``done``     : terminal frame; no payload beyond ``language``.
      - ``error``    : ``content`` holds a generic, non-sensitive error message.
    ``language`` is the detected reply language (``id``/``en``) on every frame.
    """

    type: Literal["token", "citation", "blocked", "done", "error"]
    content: str | None = None
    citations: list[CitationOut] | None = None
    language: str | None = None
