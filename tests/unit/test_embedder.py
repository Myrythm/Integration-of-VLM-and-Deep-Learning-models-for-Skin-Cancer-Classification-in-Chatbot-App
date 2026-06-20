from unittest.mock import patch, MagicMock

import pytest

from config import Settings
from utils.rag.embedder import OpenAIEmbedder


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_embedding_model="text-embedding-3-small")


def test_embed_query_returns_vector(settings: Settings) -> None:
    embedder = OpenAIEmbedder(settings)
    fake_vector = [0.1] * 1536

    with patch.object(embedder._client.embeddings, "create") as mock_create:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_vector)]
        mock_create.return_value = mock_response

        result = embedder.embed_query("apa itu melanoma?")

    assert result == fake_vector
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"
    assert call_kwargs["input"] == "apa itu melanoma?"


def test_embed_documents_batches(settings: Settings) -> None:
    embedder = OpenAIEmbedder(settings)
    fake_vectors = [[0.1] * 1536, [0.2] * 1536]

    with patch.object(embedder._client.embeddings, "create") as mock_create:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=v) for v in fake_vectors]
        mock_create.return_value = mock_response

        result = embedder.embed_documents(["text one", "text two"])

    assert result == fake_vectors
    assert mock_create.call_args.kwargs["input"] == ["text one", "text two"]
