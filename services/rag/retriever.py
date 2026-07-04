import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr


class EvidenceFilteredRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    threshold: float = 0.7

    def _filter(self, docs: list[Document]) -> list[Document]:
        kept = [d for d in docs if d.metadata.get("score", 0.0) >= self.threshold]
        # Stash the pre-filter count so the route can log retrieved-vs-kept honestly
        # (it only ever sees this retriever's post-filter output).
        for d in kept:
            d.metadata["prefilter_count"] = len(docs)
        return kept

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        _ = run_manager
        return self._filter(self.base_retriever.invoke(query))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        _ = run_manager
        return self._filter(await self.base_retriever.ainvoke(query))


class ChromaLangChainRetriever(BaseRetriever):
    """Adapter: bridges Embedder + VectorStoreProvider to LangChain retriever interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _embedder: Any = PrivateAttr(default=None)
    _vector_store: Any = PrivateAttr(default=None)
    _k: int = PrivateAttr(default=10)

    def __init__(self, embedder: Any, vector_store: Any, settings: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embedder = embedder
        self._vector_store = vector_store
        self._k = settings.rag_retrieve_k

    def _to_documents(self, results: list[dict]) -> list[Document]:
        return [
            Document(
                page_content=r["text"],
                metadata={**r["metadata"], "score": r["score"], "id": r["id"]},
            )
            for r in results
        ]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        _ = run_manager
        embedding = self._embedder.embed_query(query)
        results = self._vector_store.similarity_search(embedding, self._k)
        return self._to_documents(results)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        _ = run_manager
        embedding = await self._embedder.aembed_query(query)
        # Chroma's client is sync-only; keep it off the event loop.
        results = await asyncio.to_thread(
            self._vector_store.similarity_search, embedding, self._k
        )
        return self._to_documents(results)
