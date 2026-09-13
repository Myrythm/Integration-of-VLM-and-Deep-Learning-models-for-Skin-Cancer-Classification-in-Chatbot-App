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
    except (UnidentifiedImageError, OSError):
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
