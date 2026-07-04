from langchain_core.runnables import Runnable

from config import Settings, get_settings
from services.rag.chain import build_rag_chain
from services.rag.embedder import get_embedder
from services.rag.llm_provider import get_llm_provider
from services.rag.memory import SessionMemory
from services.rag.prompt import build_prompt_template
from services.rag.retriever import ChromaLangChainRetriever, EvidenceFilteredRetriever
from services.rag.vector_store import get_vector_store


_settings: Settings | None = None
_chain: Runnable | None = None
_memory: SessionMemory | None = None


def initialize_app_state() -> None:
    global _settings, _chain, _memory
    _settings = get_settings()
    if _settings.llm_backend == "openai" and not _settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required when llm_backend is 'openai'. "
            "Set it in .env before starting the server."
        )
    embedder = get_embedder(_settings)
    vector_store = get_vector_store(_settings)
    llm = get_llm_provider(_settings).get_chat_model()

    base_retriever = ChromaLangChainRetriever(embedder, vector_store, _settings)
    retriever = EvidenceFilteredRetriever(
        base_retriever=base_retriever,
        threshold=_settings.rag_similarity_threshold,
    )
    _chain = build_rag_chain(
        retriever,
        llm,
        build_prompt_template(),
        context_chunk_chars=_settings.rag_context_chunk_chars,
    )
    _memory = SessionMemory(
        max_turns=_settings.memory_max_turns,
        max_sessions=_settings.session_memory_max_sessions,
        ttl_seconds=_settings.session_memory_ttl_seconds,
    )


def get_chain() -> Runnable:
    if _chain is None:
        raise RuntimeError("App state not initialized. Call initialize_app_state() in lifespan.")
    return _chain


def get_memory() -> SessionMemory:
    if _memory is None:
        raise RuntimeError("App state not initialized. Call initialize_app_state() in lifespan.")
    return _memory
