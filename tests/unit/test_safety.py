from utils.rag.safety import classify_query_danger


def test_dosage_question_blocked() -> None:
    assert classify_query_danger("berapa dosis obat untuk anak?", "id") == "unsafe_dosage"
    assert classify_query_danger("what dose of methotrexate should I take?", "en") == "unsafe_dosage"


def test_medical_question_allowed() -> None:
    assert classify_query_danger("apa itu melanoma?", "id") == "safe_medical"
    assert classify_query_danger("what is melanoma?", "en") == "safe_medical"


def test_off_topic_blocked() -> None:
    assert classify_query_danger("siapa presiden indonesia?", "id") == "off_topic"
    assert classify_query_danger("what is the capital of france?", "en") == "off_topic"


def test_general_health_question_allowed() -> None:
    assert classify_query_danger("how to prevent skin aging?", "en") in {"safe_medical", "safe_general"}
