from services.rag.disclaimer import DISCLAIMERS, force_append_disclaimer


def test_disclaimers_dict_has_both_languages() -> None:
    assert "en" in DISCLAIMERS
    assert "id" in DISCLAIMERS
    assert "⚠️" in DISCLAIMERS["en"]
    assert "⚠️" in DISCLAIMERS["id"]


def test_force_appends_when_missing() -> None:
    response = "Melanoma is a serious skin cancer."
    out = force_append_disclaimer(response, "en")
    assert "Melanoma" in out
    assert DISCLAIMERS["en"] in out


def test_does_not_duplicate_when_already_present() -> None:
    response = f"Melanoma info. {DISCLAIMERS['en']}"
    out = force_append_disclaimer(response, "en")
    assert out.count(DISCLAIMERS["en"]) == 1


def test_uses_correct_language() -> None:
    out = force_append_disclaimer("Info tentang melanoma.", "id")
    assert DISCLAIMERS["id"] in out
    assert DISCLAIMERS["en"] not in out


def test_falls_back_to_english_for_unknown_language() -> None:
    out = force_append_disclaimer("Some text.", "xx")
    assert DISCLAIMERS["en"] in out
