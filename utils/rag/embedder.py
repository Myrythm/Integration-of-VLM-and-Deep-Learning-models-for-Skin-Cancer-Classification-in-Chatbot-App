from typing import Protocol

from openai import OpenAI

from config import Settings


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self._model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedder(settings: Settings) -> Embedder:
    if settings.llm_backend == "openai":
        return OpenAIEmbedder(settings)
    raise NotImplementedError(f"Embedder not implemented for backend: {settings.llm_backend}")
