# VLM Image Validation + Classifier Label-Order Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the mislabeled EfficientNetB3 output and add a GPT-4o Vision gate that rejects non-lesion images before classification.

**Architecture:** Restore the classifier's label order to match the trained model. Add a framework-agnostic async `validator` service that asks GPT-4o Vision whether an upload is a skin lesion, and wire it into `POST /api/upload` ahead of classification, failing closed on VLM errors.

**Tech Stack:** Python 3.11, FastAPI, Pydantic / pydantic-settings, `openai` (AsyncOpenAI), Pillow, pytest (asyncio_mode=auto).

## Global Constraints

- Python 3.11 only (`requires-python >=3.11,<3.12`).
- Run tests with `.venv`: `.venv\Scripts\python.exe -m pytest` (Windows). `pytest.ini` sets `asyncio_mode=auto`, so `async def test_*` needs no decorator.
- Layering rule: `services/` must NOT import FastAPI, Starlette, or `routes/`. `routes/` may import `schemas/`, `services/`, FastAPI. `schemas/` is Pydantic only.
- No AI self-attribution in commits (no `Co-Authored-By`, no "Generated with" lines).
- Classifier label order is ground truth and must be exactly: `["Karsinoma Sel Basal", "Karsinoma Sel Skuamosa", "Melanoma", "Nevus"]`.
- Default vision model: `gpt-4o`. Fail-closed on VLM failure (HTTP 503). `invalid` → 400; `valid`/`uncertain` → classify.

---

### Task 1: Fix classifier label order (Problem 2)

**Files:**
- Modify: `services/image/classifier.py:11-16` (`SKIN_CANCER_LABELS`)
- Modify: `schemas/detection.py:9-14` (`SkinCancerLabel` Literal)
- Test: `tests/unit/test_image_classifier.py`, `tests/unit/test_detection.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `SKIN_CANCER_LABELS == ["Karsinoma Sel Basal", "Karsinoma Sel Skuamosa", "Melanoma", "Nevus"]` (index → label mapping relied on implicitly by the route).

- [ ] **Step 1: Write the failing test** — add to `tests/unit/test_image_classifier.py`:

```python
def test_labels_match_model_training_order() -> None:
    # Ground truth: the order the trained skinCancer.h5 emits (from legacy/flask config.LABELS).
    assert SKIN_CANCER_LABELS == [
        "Karsinoma Sel Basal",
        "Karsinoma Sel Skuamosa",
        "Melanoma",
        "Nevus",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_image_classifier.py::test_labels_match_model_training_order -v`
Expected: FAIL — current order has "Melanoma" first.

- [ ] **Step 3: Fix the label order in the classifier**

In `services/image/classifier.py` replace the list:

```python
SKIN_CANCER_LABELS: list[str] = [
    "Karsinoma Sel Basal",
    "Karsinoma Sel Skuamosa",
    "Melanoma",
    "Nevus",
]
```

- [ ] **Step 4: Fix the Literal in the schema to match**

In `schemas/detection.py` replace the Literal members (order mirrors the classifier for readability; validation is order-independent):

```python
SkinCancerLabel = Literal[
    "Karsinoma Sel Basal",
    "Karsinoma Sel Skuamosa",
    "Melanoma",
    "Nevus",
]
```

- [ ] **Step 5: Update the classifier tests whose expected label depended on the old index-0**

In `tests/unit/test_image_classifier.py`:

`test_classify_skin_image_uses_model_prediction` — predictions `[[0.80, 0.10, 0.05, 0.05]]` now means index 0 = "Karsinoma Sel Basal":

```python
    assert result.label == "Karsinoma Sel Basal"
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence == pytest.approx(0.80, abs=1e-6)
```

`test_classify_skin_image_applies_softmax_to_logits` — logits `[[2.0, 1.0, 0.1, -1.0]]`, argmax index 0:

```python
    assert 0.0 <= result.confidence <= 1.0
    assert result.label == "Karsinoma Sel Basal"
```

(`test_classify_skin_image_picks_max_class` uses `[[0.0, 0.0, 0.0, 1.0]]` → "Nevus" at index 3, which is unchanged — leave it. `test_labels_are_four_classes` still holds — leave it.)

- [ ] **Step 6: Run the classifier + detection tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_image_classifier.py tests/unit/test_detection.py -v`
Expected: PASS (all). `test_schema_labels_match_classifier_labels` is set-based and still passes.

- [ ] **Step 7: Commit**

```bash
git add services/image/classifier.py schemas/detection.py tests/unit/test_image_classifier.py
git commit -m "fix: restore classifier label order to match trained model"
```

---

### Task 2: Add vision-model config (`openai_vision_model`, timeout)

**Files:**
- Modify: `config.py:14-16` (add fields near the other `openai_*` settings)
- Modify: `.env.example`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings.openai_vision_model: str = "gpt-4o"`, `Settings.image_validation_timeout_seconds: int = 30`.

- [ ] **Step 1: Write the failing test** — add to `tests/unit/test_config.py`:

```python
def test_settings_has_vision_defaults() -> None:
    s = Settings(openai_api_key="x")
    assert s.openai_vision_model == "gpt-4o"
    assert s.image_validation_timeout_seconds == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_config.py::test_settings_has_vision_defaults -v`
Expected: FAIL — `AttributeError` / missing fields.

- [ ] **Step 3: Add the settings fields**

In `config.py`, after `openai_embedding_model` (line 16):

```python
    openai_vision_model: str = "gpt-4o"
    image_validation_timeout_seconds: int = 30
```

- [ ] **Step 4: Document the vars in `.env.example`**

Add near the other OpenAI vars:

```bash
# Vision model used to validate that an upload is a skin lesion (GPT-4o Vision).
OPENAI_VISION_MODEL=gpt-4o
IMAGE_VALIDATION_TIMEOUT_SECONDS=30
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example tests/unit/test_config.py
git commit -m "feat: add OPENAI_VISION_MODEL and image validation timeout settings"
```

---

### Task 3: VLM validator service (Problem 1)

**Files:**
- Create: `services/image/validator.py`
- Test: `tests/unit/test_validator.py`

**Interfaces:**
- Consumes: `Settings` (from `config`), `Settings.openai_vision_model`, `Settings.openai_api_key`, `Settings.image_validation_timeout_seconds` (Task 2).
- Produces:
  - `class ValidationUnavailableError(Exception)`
  - `async def validate_skin_image(image_bytes: bytes, content_type: str, settings: Settings | None = None) -> str` returning one of `"valid" | "invalid" | "uncertain"`, raising `ValidationUnavailableError` on any API/timeout/empty-key failure.

- [ ] **Step 1: Write the failing tests** — create `tests/unit/test_validator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_validator.py -v`
Expected: FAIL — `ModuleNotFoundError: services.image.validator`.

- [ ] **Step 3: Implement the validator**

Create `services/image/validator.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_validator.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add services/image/validator.py tests/unit/test_validator.py
git commit -m "feat: add GPT-4o Vision skin-lesion validator service"
```

---

### Task 4: Wire validation into the upload route + expose `validation_status`

**Files:**
- Modify: `schemas/image.py` (add `validation_status`)
- Modify: `routes/api_routes.py` (decode pre-check, VLM gate, error mapping, response field)
- Test: `tests/integration/test_api_routes.py`

**Interfaces:**
- Consumes: `validate_skin_image`, `ValidationUnavailableError` (Task 3); `get_settings` (config); corrected labels (Task 1).
- Produces: `POST /api/upload` returns `ImageUploadResponse` with `validation_status: Literal["valid", "uncertain"]`; `invalid` → 400, VLM failure → 503, corrupt image → 400.

- [ ] **Step 1: Add `validation_status` to the response schema**

In `schemas/image.py`:

```python
from typing import Literal

from pydantic import BaseModel

from schemas.detection import DetectionResult


class ImageUploadResponse(BaseModel):
    detection: DetectionResult
    chat_session_id: str
    validation_status: Literal["valid", "uncertain"] = "valid"
    message: str = "Image processed. Continue to chat for more info."
```

- [ ] **Step 2: Write the failing integration tests**

Rewrite the top of `tests/integration/test_api_routes.py` to add a validator-mocking fixture, and add new cases. Add this fixture (keep the existing `_make_test_image_bytes` helper and `mocked_classifier` fixture):

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mocked_validator():
    """Patch the VLM validator at the route module; default verdict 'valid'."""
    with patch("routes.api_routes.validate_skin_image", new_callable=AsyncMock) as mock_v:
        mock_v.return_value = "valid"
        yield mock_v
```

Update `test_upload_image_classifies_lesion` to also request `mocked_validator` and assert the new field:

```python
def test_upload_image_classifies_lesion(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detection" in body
    assert "chat_session_id" in body
    assert body["validation_status"] == "valid"
    mocked_classifier.assert_called_once()
    mocked_validator.assert_awaited_once()
```

Add new cases:

```python
def test_upload_rejects_non_lesion_image(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    mocked_validator.return_value = "invalid"
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 400
    assert "lesi kulit" in response.json()["detail"]
    mocked_classifier.assert_not_called()


def test_upload_uncertain_still_classifies(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    mocked_validator.return_value = "uncertain"
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["validation_status"] == "uncertain"
    mocked_classifier.assert_called_once()


def test_upload_fails_closed_when_validation_unavailable(
    mocked_classifier: MagicMock, mocked_validator: AsyncMock
) -> None:
    from services.image.validator import ValidationUnavailableError

    mocked_validator.side_effect = ValidationUnavailableError("no key")
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", BytesIO(_make_test_image_bytes()), "image/png")},
    )
    assert response.status_code == 503
    mocked_classifier.assert_not_called()
```

Update the tests that reach classification but do NOT currently mock the validator so they still pass — add `mocked_validator` to their parameters: `test_upload_image_offloads_classification_to_thread`, `test_upload_image_hides_model_path_on_missing_model`, `test_upload_image_hides_internal_error_details`. Example:

```python
def test_upload_image_offloads_classification_to_thread(mocked_validator: AsyncMock) -> None:
    ...  # body unchanged
```

(`test_upload_image_rejects_unsupported_type` returns 400 before validation — leave as is. `test_upload_image_rejects_corrupt_image_bytes` now returns 400 from the new local decode check before the VLM — leave its assertions unchanged; it does not need the validator mock. `test_upload_image_rejects_oversize_file` is unaffected.)

- [ ] **Step 3: Run the integration tests to verify the new ones fail**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_api_routes.py -v`
Expected: the three new tests FAIL (validation not wired); `validation_status` assertion fails.

- [ ] **Step 4: Wire validation into the route**

Rewrite `routes/api_routes.py` `upload_image` to add the decode pre-check and VLM gate. Full file:

```python
import asyncio
import io
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from config import get_settings
from schemas.image import ImageUploadResponse
from services.image.classifier import classify_skin_image
from services.image.validator import ValidationUnavailableError, validate_skin_image

router = APIRouter(prefix="/api", tags=["image"])
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB

_NOT_A_LESION_MSG = (
    "Gambar yang diunggah bukan gambar lesi kulit. "
    "/ The uploaded image is not a skin lesion."
)
_VALIDATION_UNAVAILABLE_MSG = (
    "Validasi gambar tidak tersedia saat ini. Silakan coba lagi nanti. "
    "/ Image validation is unavailable right now. Please try again later."
)


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(CHUNK_SIZE):
        total_size += len(chunk)
        if total_size > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 10 MB)")
        chunks.append(chunk)
    contents = b"".join(chunks)

    # Cheap local decode check: reject non-image bytes before spending a VLM call.
    try:
        Image.open(io.BytesIO(contents)).verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    settings = get_settings()
    try:
        validation_status = await validate_skin_image(contents, file.content_type, settings)
    except ValidationUnavailableError:
        logger.exception("Image validation unavailable")
        raise HTTPException(status_code=503, detail=_VALIDATION_UNAVAILABLE_MSG)

    if validation_status == "invalid":
        raise HTTPException(status_code=400, detail=_NOT_A_LESION_MSG)

    try:
        detection = await asyncio.to_thread(classify_skin_image, contents)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    except FileNotFoundError:
        logger.exception("Classification model file not found")
        raise HTTPException(status_code=503, detail="Classification model unavailable")
    except Exception:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail="Internal classification error")

    return ImageUploadResponse(
        detection=detection,
        chat_session_id=str(uuid.uuid4()),
        validation_status=validation_status,
    )
```

- [ ] **Step 5: Run the full upload test file to verify all pass**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_api_routes.py -v`
Expected: PASS (all, including the three new cases).

- [ ] **Step 6: Commit**

```bash
git add schemas/image.py routes/api_routes.py tests/integration/test_api_routes.py
git commit -m "feat: gate image upload with GPT-4o Vision lesion validation"
```

---

### Task 5: Full-suite regression + docs sync

**Files:**
- Modify: `CLAUDE.md` (note the new validation step in the request flow), `README.md` (optional env var mention)
- Test: whole suite

**Interfaces:**
- Consumes: everything above.
- Produces: green test suite; docs reflect the validation gate.

- [ ] **Step 1: Run the whole suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS (the `tests/eval/` harness is not part of the normal run and needs a live key — it is excluded by default).

- [ ] **Step 2: Update `CLAUDE.md` request-flow bullet**

In the `## Architecture` → request flow, update the upload bullet to note validation runs first:

```markdown
- `POST /api/upload` → `routes/api_routes.py` decodes the image, runs
  `services/image/validator.validate_skin_image` (GPT-4o Vision; `invalid` → 400,
  VLM failure → 503 fail-closed), then `services/image/classifier.classify_skin_image`
  (via `asyncio.to_thread`) returns a `DetectionResult` + `validation_status` + a new
  `chat_session_id`.
```

- [ ] **Step 3: Mention the env var in `README.md`**

Add a row to the env-var table:

```markdown
| `OPENAI_VISION_MODEL` | `gpt-4o` | Vision model that validates an upload is a skin lesion. |
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document GPT-4o Vision validation gate on image upload"
```

---

## Self-Review

**Spec coverage:**
- Problem 2 label fix → Task 1 ✓ (classifier + schema + tests).
- VLM validator service, fail-closed, unrecognized→uncertain → Task 3 ✓.
- Config `openai_vision_model` default `gpt-4o` + timeout + `.env.example` → Task 2 ✓.
- Route wiring: decode pre-check, invalid→400, error→503, valid/uncertain→classify → Task 4 ✓.
- `validation_status` in response → Task 4 ✓.
- Tests (unit validator, detection order, integration upload cases) → Tasks 1, 3, 4 ✓.
- Docs → Task 5 ✓.

**Placeholder scan:** No TBD/TODO; all code steps show full code. ✓

**Type consistency:** `validate_skin_image(image_bytes, content_type, settings)` and `ValidationUnavailableError` are defined in Task 3 and consumed with the same names/signature in Task 4. `validation_status` typed `Literal["valid", "uncertain"]` consistently in schema (Task 4) and route return (Task 4); the validator may also return `"invalid"`, which the route intercepts before constructing the response. Label list identical across Task 1 uses. ✓
