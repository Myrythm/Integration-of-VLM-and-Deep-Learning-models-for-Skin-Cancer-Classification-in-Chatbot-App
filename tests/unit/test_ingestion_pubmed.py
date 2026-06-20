import json
from pathlib import Path

from utils.rag.ingestion import parse_pubmed_json


def test_parse_pubmed_json(tmp_path: Path) -> None:
    pubmed_file = tmp_path / "pmid_123.json"
    pubmed_file.write_text(json.dumps({
        "pmid": "123",
        "title": "Test Article",
        "abstract": "This is the abstract body.",
        "journal": "Test Journal",
        "publish_date": "2023-06",
        "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
    }))
    chunks = parse_pubmed_json(pubmed_file)
    assert len(chunks) >= 1
    assert chunks[0]["metadata"]["source"] == "pubmed"
    assert chunks[0]["metadata"]["pmid"] == "123"
    assert "abstract body" in chunks[0]["text"]
