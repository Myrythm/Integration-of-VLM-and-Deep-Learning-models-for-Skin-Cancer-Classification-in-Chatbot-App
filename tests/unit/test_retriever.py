from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import Settings
from services.rag.retriever import ChromaLangChainRetriever, EvidenceFilteredRetriever


class FakeRetriever(BaseRetriever):
    docs: list[dict] = []

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return [
            Document(page_content=d["text"], metadata={**d["metadata"], "score": d["score"]})
            for d in self.docs
        ]


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeStore:
    def similarity_search(self, embedding: list[float], k: int) -> list[dict]:
        return [
            {"id": "1", "text": "melanoma info", "metadata": {"source": "aad"}, "score": 0.9},
        ]


def test_chroma_langchain_retriever_bridges_and_sets_score() -> None:
    settings = Settings(openai_api_key="x", rag_retrieve_k=3)
    r = ChromaLangChainRetriever(_FakeEmbedder(), _FakeStore(), settings)
    docs = r.invoke("q")
    assert len(docs) == 1
    assert docs[0].page_content == "melanoma info"
    assert docs[0].metadata["score"] == 0.9
    assert docs[0].metadata["id"] == "1"
    assert docs[0].metadata["source"] == "aad"


def test_drops_chunks_below_threshold() -> None:
    docs = [
        {"text": "high relevance", "metadata": {"id": "1"}, "score": 0.9},
        {"text": "low relevance", "metadata": {"id": "2"}, "score": 0.5},
        {"text": "borderline", "metadata": {"id": "3"}, "score": 0.69},
    ]
    fake = FakeRetriever(docs=docs)
    ef = EvidenceFilteredRetriever(base_retriever=fake, threshold=0.7)

    out = ef.invoke("test query")
    ids = [d.metadata["id"] for d in out]
    assert ids == ["1"]


def test_filter_stashes_prefilter_count() -> None:
    docs = [
        {"text": "a", "metadata": {"id": "1"}, "score": 0.9},
        {"text": "b", "metadata": {"id": "2"}, "score": 0.1},
        {"text": "c", "metadata": {"id": "3"}, "score": 0.8},
    ]
    ef = EvidenceFilteredRetriever(base_retriever=FakeRetriever(docs=docs), threshold=0.5)
    out = ef.invoke("q")
    assert len(out) == 2
    assert all(d.metadata["prefilter_count"] == 3 for d in out)


def test_includes_chunks_at_or_above_threshold() -> None:
    docs = [
        {"text": "ok", "metadata": {"id": "1"}, "score": 0.7},
        {"text": "ok", "metadata": {"id": "2"}, "score": 0.71},
    ]
    fake = FakeRetriever(docs=docs)
    ef = EvidenceFilteredRetriever(base_retriever=fake, threshold=0.7)
    out = ef.invoke("test")
    assert len(out) == 2


def test_empty_input_returns_empty() -> None:
    fake = FakeRetriever(docs=[])
    ef = EvidenceFilteredRetriever(base_retriever=fake, threshold=0.7)
    assert ef.invoke("test") == []


class _NoScoreRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return [Document(page_content="no score", metadata={"id": "1"})]


def test_missing_score_fails_closed() -> None:
    ef = EvidenceFilteredRetriever(base_retriever=_NoScoreRetriever(), threshold=0.3)
    assert ef.invoke("test") == []


class _AsyncOnlyEmbedder:
    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("sync embed path used; async path expected")

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


async def test_chroma_retriever_ainvoke_uses_async_embed() -> None:
    settings = Settings(openai_api_key="x", rag_retrieve_k=3)
    r = ChromaLangChainRetriever(_AsyncOnlyEmbedder(), _FakeStore(), settings)
    docs = await r.ainvoke("q")
    assert len(docs) == 1
    assert docs[0].metadata["score"] == 0.9


class _AsyncOnlyBase(BaseRetriever):
    docs: list[dict] = []

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise AssertionError("sync path used; async path expected")

    async def _aget_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return [
            Document(page_content=d["text"], metadata={**d["metadata"], "score": d["score"]})
            for d in self.docs
        ]


async def test_evidence_filter_ainvoke_uses_async_base() -> None:
    base = _AsyncOnlyBase(
        docs=[
            {"text": "hi", "metadata": {"id": "1"}, "score": 0.9},
            {"text": "lo", "metadata": {"id": "2"}, "score": 0.1},
        ]
    )
    ef = EvidenceFilteredRetriever(base_retriever=base, threshold=0.5)
    out = await ef.ainvoke("q")
    assert [d.metadata["id"] for d in out] == ["1"]
