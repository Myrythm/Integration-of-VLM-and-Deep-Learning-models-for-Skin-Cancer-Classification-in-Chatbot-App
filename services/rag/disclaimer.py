DISCLAIMERS: dict[str, str] = {
    "en": (
        "⚠️ I'm an AI assistant for education only. "
        "This is not medical advice. "
        "Always consult a qualified dermatologist for diagnosis and treatment."
    ),
    "id": (
        "⚠️ Saya asisten AI untuk edukasi saja. "
        "Ini bukan nasihat medis. "
        "Selalu konsultasi dengan dokter spesialis kulit untuk diagnosis dan pengobatan."
    ),
}


def force_append_disclaimer(response: str, language: str) -> str:
    """Ensure the response ends with the disclaimer for the given language.
    If the disclaimer is already present, do not duplicate. Idempotent.
    """
    disclaimer = DISCLAIMERS.get(language, DISCLAIMERS["en"])
    if disclaimer in response:
        return response
    if not response.strip():
        return disclaimer
    return f"{response.rstrip()}\n\n{disclaimer}"
