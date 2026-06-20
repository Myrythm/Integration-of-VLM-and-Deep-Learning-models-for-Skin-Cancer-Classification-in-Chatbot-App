import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from schemas.image import ImageUploadResponse
from services.image.classifier import classify_skin_image

router = APIRouter(prefix="/api", tags=["image"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)) -> ImageUploadResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    detection = classify_skin_image(contents)
    return ImageUploadResponse(
        detection=detection,
        chat_session_id=str(uuid.uuid4()),
    )
