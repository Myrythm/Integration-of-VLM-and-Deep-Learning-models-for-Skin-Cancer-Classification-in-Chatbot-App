from schemas.detection import DetectionResult


def classify_skin_image(image_bytes: bytes) -> DetectionResult:
    """Stub: real EfficientNetB3 model will replace this. Returns a mock
    classification so the rest of the pipeline can be built and tested.
    """
    return DetectionResult(
        label="benign_nevus",
        confidence=0.87,
    )
