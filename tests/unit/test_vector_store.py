import tempfile
from pathlib import Path

import pytest

from config import Settings
from services.rag.vector_store import ChromaProvider


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


def test_delete_collection_wipes_data_and_allows_reuse(settings: Settings) -> None:
    provider = ChromaProvider(settings)
    chunk = {
        "id": "1",
        "text": "melanoma is a serious skin cancer",
        "metadata": {"source": "aad", "url": "https://aad.org/x", "title": "Melanoma"},
    }
    provider.upsert([chunk], [[0.1] * 1536])
    assert len(provider.similarity_search([0.1] * 1536, k=5)) == 1

    provider.delete_collection()
    assert provider.similarity_search([0.1] * 1536, k=5) == []

    # Collection is recreated, so subsequent upserts still work.
    provider.upsert([chunk], [[0.2] * 1536])
    assert len(provider.similarity_search([0.2] * 1536, k=5)) == 1
