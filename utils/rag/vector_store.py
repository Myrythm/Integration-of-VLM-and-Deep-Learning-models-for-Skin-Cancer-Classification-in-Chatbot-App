from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import Settings


class VectorStoreProvider(Protocol):
    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> None: ...
    def similarity_search(self, query_embedding: list[float], k: int) -> list[dict]: ...


class ChromaProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )

    def similarity_search(self, query_embedding: list[float], k: int) -> list[dict]:
        results = self._collection.query(query_embeddings=[query_embedding], n_results=k)

        ids = results["ids"][0] if results["ids"] else []
        docs = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        out = []
        for doc_id, doc_text, meta, dist in zip(ids, docs, metadatas, distances):
            out.append({
                "id": doc_id,
                "text": doc_text,
                "metadata": meta,
                "score": 1.0 - dist,
            })
        return out


def get_vector_store(settings: Settings) -> VectorStoreProvider:
    if settings.vector_store_backend == "chroma":
        return ChromaProvider(settings)
    if settings.vector_store_backend == "pinecone":
        raise NotImplementedError("PineconeProvider not yet implemented (Phase 3 stub)")
    raise ValueError(f"Unknown vector store backend: {settings.vector_store_backend}")
