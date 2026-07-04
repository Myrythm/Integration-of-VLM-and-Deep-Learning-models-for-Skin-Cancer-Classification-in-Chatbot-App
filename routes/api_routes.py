import asyncio
import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from schemas.image import ImageUploadResponse
from services.image.classifier import classify_skin_image

router = APIRouter(prefix="/api", tags=["image"])
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB


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
    )
