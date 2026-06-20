import re

_DOSAGE_PATTERNS = [
    re.compile(r"\bdos[ie]s?\b", re.IGNORECASE),
    re.compile(r"\bdose\b", re.IGNORECASE),
    re.compile(r"\bberapa (mg|ml|mcg|tablet|kapsul)\b", re.IGNORECASE),
    re.compile(r"\bmg\b|\bml\b|\bmcg\b", re.IGNORECASE),
    re.compile(r"\bhow much (should i|to) take\b", re.IGNORECASE),
    re.compile(r"\bshould i take\b", re.IGNORECASE),
]

_OFF_TOPIC_KEYWORDS_ID = [
    "presiden", "politik", "sepak bola", "musik", "film", "resep masakan",
    "cuaca", "ekonomi", "saham", "kripto",
]
_OFF_TOPIC_KEYWORDS_EN = [
    "president", "politics", "football", "soccer", "music", "movie", "recipe",
    "weather", "economy", "stock", "crypto", "bitcoin", "capital",
]


def classify_query_danger(query: str, language: str) -> str:
    """Heuristic classifier. Returns one of: safe_medical, safe_general,
    unsafe_dosage, off_topic. For production, layer with a small LLM
    classifier for ambiguous cases.
    """
    q = query.lower().strip()
    if not q:
        return "off_topic"

    for pat in _DOSAGE_PATTERNS:
        if pat.search(q):
            return "unsafe_dosage"

    off_topic_words = _OFF_TOPIC_KEYWORDS_ID if language == "id" else _OFF_TOPIC_KEYWORDS_EN
    if any(w in q for w in off_topic_words):
        return "off_topic"

    skin_keywords = [
        "kulit", "skin", "kanker", "cancer", "melanoma", "lesi", "lesion",
        "mole", "tahi lalat", "bintik", "spot", "dermatolog", "dermatitis",
        "jerawat", "acne", "gatal", "itch", "rash", "biopsy", "dermatology",
    ]
    if any(w in q for w in skin_keywords):
        return "safe_medical"

    return "safe_general"
