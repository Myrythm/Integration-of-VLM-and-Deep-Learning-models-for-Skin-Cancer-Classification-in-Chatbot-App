from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class EvidenceFilteredRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    threshold: float = 0.7

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        _ = run_manager
        docs = self.base_retriever.invoke(query)
        return [d for d in docs if d.metadata.get("score", 1.0) >= self.threshold]
