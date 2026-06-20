from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0

_SUPPORTED = {"id", "en"}


def detect_language(text: str) -> str:
    """Return 'id' or 'en'. Defaults to 'en' on empty/short/ambiguous input."""
    if not text or len(text.strip()) < 4:
        return "en"
    try:
        candidates = detect_langs(text)
    except Exception:
        return "en"
    if not candidates:
        return "en"
    top = candidates[0]
    if top.lang in _SUPPORTED and top.prob >= 0.5:
        return top.lang
    return "en"
