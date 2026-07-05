# Design: VLM Image Validation + Classifier Label-Order Fix

**Date:** 2026-07-05
**Target project:** `Integration-of-GPT-4o-Vision-and-EfficientNetB3-for-Skin-Cancer-Classification-in-Chatbot-App`
**Status:** Approved (pending spec review)

## Summary

Two independent defects in the FastAPI app, fixed together because they both live in
the image-upload path (`routes/api_routes.py` → `services/image/`):

1. **No image validation.** Any uploaded image is fed straight to the EfficientNetB3
   classifier, so non-lesion photos (faces, objects, scenery) get a confident but
   meaningless skin-cancer label. The legacy Flask app gated this with a GPT-4o Vision
   check (`utils/validation.py`) that never made it into the FastAPI rewrite.

2. **Wrong classifier labels.** The FastAPI rewrite reordered `SKIN_CANCER_LABELS`
   relative to the model's training order, so three of the four classes are mislabeled.
   The model and preprocessing are otherwise correct.

## Problem 2 — label-order fix

### Root cause

The trained model (`model/skinCancer.h5`) emits class probabilities in the order the
legacy app declared, which is the ground truth:

```python
# legacy/flask config.py  (matches the model's output indices)
LABELS = ['Karsinoma Sel Basal', 'Karsinoma Sel Skuamosa', 'Melanoma', 'Nevus']
```

The FastAPI code reordered them, putting Melanoma at index 0:

```python
# services/image/classifier.py  (WRONG)
SKIN_CANCER_LABELS = ["Melanoma", "Karsinoma Sel Basal", "Karsinoma Sel Skuamosa", "Nevus"]
```

Result: `argmax` index 0 is really *Karsinoma Sel Basal* but is reported as *Melanoma*,
etc. Only *Nevus* (index 3 in both) stays correct.

Preprocessing is **not** the cause. Legacy (`keras.preprocessing.image.img_to_array`)
and current (`np.array(img, dtype=np.float32)`) both feed raw 0–255 RGB at 224×224 with
no `preprocess_input`; they are equivalent, and the model has its normalization baked in
(legacy works without external normalization). Preprocessing is left unchanged.

### Fix

Restore the legacy order in **both** places that hardcode it:

- `services/image/classifier.py` → `SKIN_CANCER_LABELS`
- `schemas/detection.py` → `SkinCancerLabel` Literal

The order becomes:

```python
["Karsinoma Sel Basal", "Karsinoma Sel Skuamosa", "Melanoma", "Nevus"]
```

`tests/unit/test_detection.py` already asserts these two lists stay in sync; its expected
value is updated to the corrected order.

## Problem 1 — GPT-4o Vision validation

### New component: `services/image/validator.py`

Framework-agnostic module (obeys the layering rule: no FastAPI / Starlette / `routes/`
imports; may import `openai`, `config`, `schemas`).

```
async def validate_skin_image(
    image_bytes: bytes,
    content_type: str,
    settings: Settings | None = None,
) -> str            # returns "valid" | "invalid" | "uncertain"
```

Behavior:

- Uses `openai.AsyncOpenAI(api_key=settings.openai_api_key)` — native async, no
  `asyncio.to_thread`.
- Model: new `settings.openai_vision_model` (default `gpt-4o`).
- Encodes the in-memory bytes as base64 into a `data:{content_type};base64,{...}` URL.
  **No temp file** — the legacy app wrote uploads to disk; we don't need to.
- `temperature=0`, small `max_tokens` (≈10), and a request timeout from
  `settings.image_validation_timeout_seconds` (default 30s).
- System + user prompt ported verbatim from legacy `utils/validation.py` (dermatology
  imaging specialist; respond only with `valid` / `invalid` / `uncertain`).
- Normalizes the reply (strip, lowercase). If it exactly matches one of the three
  verdicts, return it. If it is unrecognized, log a warning and return `"uncertain"`
  (lets the image through — a parsing hiccup should not reject a real lesion).
- On any OpenAI SDK / network / timeout error, or an empty API key, raise a custom
  `ValidationUnavailableError` (defined in this module). This is the fail-closed signal.

### Wiring: `routes/api_routes.py`

Insert validation after the bytes are fully read and before `classify_skin_image`:

1. `result = await validate_skin_image(contents, file.content_type, settings)`
2. `result == "invalid"` → `HTTPException(400, detail=<bilingual message>)`; do not classify.
3. `ValidationUnavailableError` → `HTTPException(503, detail=<bilingual message>)`; do not classify.
4. `result in ("valid", "uncertain")` → proceed to the existing classification path.

`settings` is obtained via `config.get_settings()` (consistent with the rest of the app).

Bilingual rejection message (Indonesian primary, English secondary), e.g.:
`"Gambar yang diunggah bukan gambar lesi kulit. / The uploaded image is not a skin lesion."`

### Config

`config.py` `Settings`:

```python
openai_vision_model: str = "gpt-4o"
image_validation_timeout_seconds: int = 30
```

`.env.example`: document `OPENAI_VISION_MODEL` (and the timeout).

### Schema

`schemas/image.py` `ImageUploadResponse`:

```python
validation_status: Literal["valid", "uncertain"]   # only these reach a 200 response
```

Exposed so the UI can surface an "uncertain" advisory to the user.

## Error-handling summary

| Verdict / event                                  | HTTP | Classifies? |
|--------------------------------------------------|------|-------------|
| `valid` / `uncertain`                            | 200  | yes         |
| `invalid`                                         | 400  | no          |
| VLM call fails (API error, timeout, empty key)   | 503  | no          |
| Existing image-decode / model errors             | as today (400 / 503 / 500) | — |

## Testing

- **unit — `tests/unit/test_validator.py`** (new): mock `AsyncOpenAI`; cover
  `valid`, `invalid`, `uncertain`, unrecognized-reply→`uncertain`, and
  API-error→`ValidationUnavailableError` (also empty-key→error).
- **unit — `tests/unit/test_detection.py`**: update expected label order.
- **integration — upload route** (`tests/integration/`): with the validator mocked —
  `invalid`→400, `uncertain`→classifies (200), `ValidationUnavailableError`→503, and the
  `validation_status` field is present on 200. Existing upload-flow tests get the
  validator mock added so they continue to pass.

## Out of scope

- Changing the classifier model, preprocessing math, or input size.
- Retraining or re-exporting the `.h5` model.
- Caching or rate-limiting VLM calls.
- UI template changes beyond consuming the new `validation_status` field (can follow).

## Files touched

- `services/image/classifier.py` (label order)
- `schemas/detection.py` (label order)
- `services/image/validator.py` (new)
- `routes/api_routes.py` (wire validation, error mapping)
- `config.py` (`openai_vision_model`, timeout)
- `schemas/image.py` (`validation_status`)
- `.env.example` (document new vars)
- `tests/unit/test_validator.py` (new), `tests/unit/test_detection.py`,
  `tests/integration/` upload tests
