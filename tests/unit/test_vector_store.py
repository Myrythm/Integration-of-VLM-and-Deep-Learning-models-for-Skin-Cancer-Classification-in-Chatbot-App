import tempfile
from pathlib import Path

import pytest

from config import Settings
from utils.rag.vector_store import ChromaProvider


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test",
        chroma_path=str(tmp_path / "chroma_test"),
        chroma_collection="test_coll",
    )


def test_upsert_and_similarity_search_roundtrip(settings: Settings) -> None:
    provider = ChromaProvider(settings)
    chunks = [
        {
            "id": "1",
            "text": "melanoma is a serious skin cancer",
            "metadata": {"source": "aad", "url": "https://aad.org/x", "title": "Melanoma"},
        },
        {
            "id": "2",
            "text": "basal cell carcinoma is the most common skin cancer",
            "metadata": {"source": "medlineplus", "url": "https://medlineplus.gov/y", "title": "BCC"},
        },
    ]
    embeddings = [[0.1] * 1536, [0.9] * 1536]

    provider.upsert(chunks, embeddings)

    query_emb = [0.15] * 1536
    results = provider.similarity_search(query_emb, k=2)

    assert len(results) == 2
    assert all("text" in r and "metadata" in r and "score" in r for r in results)
    assert results[0]["id"] == "1"
    assert results[0]["metadata"]["source"] == "aad"
