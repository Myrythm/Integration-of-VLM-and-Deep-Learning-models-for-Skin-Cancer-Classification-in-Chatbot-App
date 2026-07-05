import base64
import logging

from openai import AsyncOpenAI

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_VERDICTS = {"valid", "invalid", "uncertain"}

_SYSTEM_PROMPT = """
You are a dermatology imaging specialist trained to analyze skin lesions for signs of malignancy,
including both cancerous (melanoma, basal cell carcinoma, squamous cell carcinoma) and benign (nevi, seborrheic keratosis)
conditions, as well as other non-cancerous skin diseases (acne, eczema, psoriasis, warts, etc.)

Based on the image provided, classify it according to these rules:
1. If the image clearly shows possible skin cancer (e.g., irregular borders, color variation, rapid growth) or a benign lesion that requires monitoring (e.g., atypical nevus), respond with 'valid'.
2. If it is clearly a wound (e.g., a healing injury, scar, scar tissue) or scar unrelated to skin cancer (e.g., acne, eczema, psoriasis, warts), or if the image is unrelated or lacks sufficient detail, respond with 'invalid'.
3. If the image is ambiguous (e.g., early-stage lesions, low-resolution images) or does not provide enough information for a confident determination, respond with 'uncertain'.

Respond with only 'valid', 'invalid', or 'uncertain'."""


class ValidationUnavailableError(Exception):
    """Raised when the vision validation cannot be completed (API/network/timeout/config)."""


async def validate_skin_image(
    image_bytes: bytes,
    content_type: str,
    settings: Settings | None = None,
) -> str:
    """Ask GPT-4o Vision whether the image is a skin lesion.

    Returns one of "valid" | "invalid" | "uncertain". An unrecognized reply is
    treated as "uncertain". Raises ValidationUnavailableError on any failure so
    the caller can fail closed.
    """
    if settings is None:
        settings = get_settings()
    if not settings.openai_api_key:
        raise ValidationUnavailableError("OpenAI API key is not configured")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64}"

    try:
        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.image_validation_timeout_seconds,
        )
        response = await client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is this image possibly to be a skin cancer lesion, either benign or malignant?",
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=10,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on any SDK/network error
        logger.exception("Vision validation call failed")
        raise ValidationUnavailableError(str(exc)) from exc

    raw = (response.choices[0].message.content or "").strip().lower()
    if raw in _VERDICTS:
        return raw
    logger.warning("Unrecognized vision validation reply %r; treating as uncertain", raw)
    return "uncertain"
