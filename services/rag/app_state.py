from langchain_core.runnables import Runnable

from config import Settings, get_settings
from services.rag.chain import build_rag_chain
from services.rag.embedder import get_embedder
from services.rag.llm_provider import get_llm_provider
from services.rag.memory import SessionMemory
from services.rag.prompt import build_prompt_template
from services.rag.retriever import EvidenceFilteredRetriever
from services.rag.vector_store import get_vector_store


_settings: Settings | None = None
_chain: Runnable | None = None
_memory: SessionMemory | None = None


def initialize_app_state() -> None:
    global _settings, _chain, _memory
    _settings = get_settings()
    embedder = get_embedder(_settings)
    vector_store = get_vector_store(_settings)
    llm = get_llm_provider(_settings).get_streaming_chat_model()

    base_retriever = _ChromaLangChainRetriever(embedder, vector_store, _settings)
    retriever = EvidenceFilteredRetriever(
        base_retriever=base_retriever,
        threshold=_settings.rag_similarity_threshold,
    )
    _chain = build_rag_chain(retriever, llm, build_prompt_template())
    _memory = SessionMemory(max_turns=6)


def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("App state not initialized. Call initialize_app_state() in lifespan.")
    return _settings


def get_chain() -> Runnable:
    if _chain is None:
        raise RuntimeError("App state not initialized. Call initialize_app_state() in lifespan.")
    return _chain


def get_memory() -> SessionMemory:
    if _memory is None:
        raise RuntimeError("App state not initialized. Call initialize_app_state() in lifespan.")
    return _memory


class _ChromaLangChainRetriever:
    """Adapter: bridges Embedder + VectorStoreProvider to LangChain retriever interface."""

    def __init__(self, embedder, vector_store, settings) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._k = settings.rag_retrieve_k

    def invoke(self, query: str, config=None) -> list:
        from langchain_core.documents import Document
        embedding = self._embedder.embed_query(query)
        results = self._vector_store.similarity_search(embedding, self._k)
        return [
            Document(
                page_content=r["text"],
                metadata={**r["metadata"], "score": r["score"], "id": r["id"]},
            )
            for r in results
        ]
