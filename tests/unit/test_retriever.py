from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from services.rag.retriever import EvidenceFilteredRetriever


class FakeRetriever(BaseRetriever):
    docs: list[dict] = []

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return [
            Document(page_content=d["text"], metadata={**d["metadata"], "score": d["score"]})
            for d in self.docs
        ]


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
