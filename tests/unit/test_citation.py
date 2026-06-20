from services.rag.citation import extract_citations, format_for_ui


def test_extracts_single_citation() -> None:
    assert extract_citations("Melanoma is serious [1].") == [1]


def test_extracts_multiple_citations() -> None:
    assert extract_citations("See [1] and [3] for details.") == [1, 3]


def test_extracts_unique_sorted() -> None:
    assert extract_citations("See [3], [1], [2].") == [1, 2, 3]


def test_no_citations_returns_empty() -> None:
    assert extract_citations("No citations here.") == []


def test_handles_multi_digit() -> None:
    assert extract_citations("Reference [12] and [3].") == [3, 12]


def test_format_for_ui_uses_metadata() -> None:
    chunks = [
        {"id": "1", "text": "...", "metadata": {"title": "Melanoma", "url": "https://aad.org/x", "source": "aad"}, "score": 0.92},
        {"id": "2", "text": "...", "metadata": {"title": "BCC", "url": "https://medlineplus.gov/y", "source": "medlineplus"}, "score": 0.81},
    ]
    cited_nums = [1, 2]
    ui = format_for_ui(chunks, cited_nums)
    assert len(ui) == 2
    assert ui[0]["number"] == 1
    assert ui[0]["title"] == "Melanoma"
    assert ui[0]["url"] == "https://aad.org/x"
    assert ui[0]["source"] == "aad"
    assert ui[1]["number"] == 2
