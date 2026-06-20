import pytest

from config import Settings
from utils.rag.vector_store import ChromaProvider, PineconeProvider, get_vector_store


def test_chroma_backend_returns_chroma_provider(tmp_path) -> None:
    settings = Settings(
        openai_api_key="test",
        chroma_path=str(tmp_path / "chroma"),
    )
    provider = get_vector_store(settings)
    assert isinstance(provider, ChromaProvider)


def test_pinecone_backend_raises_not_implemented() -> None:
    settings = Settings(openai_api_key="test", vector_store_backend="pinecone")
    with pytest.raises(NotImplementedError, match="PineconeProvider"):
        get_vector_store(settings)


def test_unknown_backend_raises() -> None:
    settings = Settings(openai_api_key="test", vector_store_backend="weaviate")
    with pytest.raises(ValueError, match="Unknown vector store"):
        get_vector_store(settings)
