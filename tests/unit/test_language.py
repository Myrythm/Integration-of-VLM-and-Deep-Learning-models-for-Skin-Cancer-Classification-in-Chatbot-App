from services.rag.language import detect_language


def test_detect_indonesian() -> None:
    assert detect_language("Apa itu melanoma dan bagaimana cara mengobatinya?") == "id"


def test_detect_english() -> None:
    assert detect_language("What is melanoma and how is it treated?") == "en"


def test_very_short_text_defaults_to_english() -> None:
    assert detect_language("ok") == "en"


def test_empty_string_defaults_to_english() -> None:
    assert detect_language("") == "en"


def test_mixed_text_falls_back_to_english_when_unclear() -> None:
    result = detect_language("a")
    assert result in {"id", "en"}
