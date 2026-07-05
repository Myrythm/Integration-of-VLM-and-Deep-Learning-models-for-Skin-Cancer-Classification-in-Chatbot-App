from unittest.mock import AsyncMock, patch

import pytest

from config import Settings
from services.image.validator import (
    ValidationUnavailableError,
    validate_skin_image,
)

IMG = b"fake-image-bytes"


def _settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_vision_model="gpt-4o")


def _mock_client_returning(text: str) -> AsyncMock:
    """Build an AsyncOpenAI-shaped mock whose chat completion returns `text`."""
    client = AsyncMock()
    message = type("Msg", (), {"content": text})
    choice = type("Choice", (), {"message": message})
    completion = type("Completion", (), {"choices": [choice]})
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


@pytest.mark.parametrize("reply,expected", [
    ("valid", "valid"),
    ("VALID", "valid"),
    (" invalid ", "invalid"),
    ("uncertain", "uncertain"),
])
async def test_returns_recognized_verdict(reply: str, expected: str) -> None:
    client = _mock_client_returning(reply)
    with patch("services.image.validator.AsyncOpenAI", return_value=client):
        result = await validate_skin_image(IMG, "image/png", _settings())
    assert result == expected


async def test_unrecognized_reply_defaults_to_uncertain() -> None:
    client = _mock_client_returning("I cannot tell from this image")
    with patch("services.image.validator.AsyncOpenAI", return_value=client):
        result = await validate_skin_image(IMG, "image/png", _settings())
    assert result == "uncertain"


async def test_api_error_raises_validation_unavailable() -> None:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("services.image.validator.AsyncOpenAI", return_value=client):
        with pytest.raises(ValidationUnavailableError):
            await validate_skin_image(IMG, "image/png", _settings())


async def test_empty_api_key_raises_validation_unavailable() -> None:
    with pytest.raises(ValidationUnavailableError):
        await validate_skin_image(IMG, "image/png", Settings(openai_api_key=""))


async def test_sends_base64_data_url_with_content_type() -> None:
    client = _mock_client_returning("valid")
    with patch("services.image.validator.AsyncOpenAI", return_value=client):
        await validate_skin_image(IMG, "image/jpeg", _settings())
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    # image url is embedded in the user message content
    user_msg = kwargs["messages"][-1]["content"]
    image_part = next(p for p in user_msg if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
