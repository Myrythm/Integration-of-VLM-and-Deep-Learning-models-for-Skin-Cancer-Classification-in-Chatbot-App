import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from services.rag.chain import build_rag_chain, format_docs
from services.rag.memory import SessionMemory
from services.rag.prompt import build_prompt_template


class _AsyncOnlyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        raise AssertionError("sync retrieve used; async path expected")

    async def _aget_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="ctx", metadata={"source": "aad"})]


async def test_chain_retrieve_step_is_async_and_named() -> None:
    llm = RunnableLambda(lambda _prompt_value: "answer")
    chain = build_rag_chain(_AsyncOnlyRetriever(), llm, build_prompt_template())

    names = []
    async for event in chain.astream_events(
        {
            "question": "q",
            "language": "en",
            "chat_history": [],
            "detection": "",
            "disclaimer": "D",
        },
        version="v2",
    ):
        names.append(event.get("name"))

    assert "retrieve" in names


def test_format_docs_numbers_chunks() -> None:
    docs = [
        Document(page_content="text one", metadata={"title": "A"}),
        Document(page_content="text two", metadata={"title": "B"}),
    ]
    out = format_docs(docs)
    assert "[1] text one" in out
    assert "[2] text two" in out


def test_format_docs_respects_max_chars() -> None:
    docs = [Document(page_content="x" * 5000, metadata={"title": "A"})]
    out = format_docs(docs, max_chars=100)
    assert "x" * 100 in out
    assert "x" * 101 not in out


def test_build_rag_chain_returns_runnable() -> None:
    fake_retriever = MagicMock()
    fake_llm = MagicMock(spec=BaseChatModel)
    prompt = build_prompt_template()
    chain = build_rag_chain(retriever=fake_retriever, llm=fake_llm, prompt=prompt)
    assert chain is not None
    assert hasattr(chain, "invoke")
    assert hasattr(chain, "stream")


@pytest.mark.asyncio
async def test_session_memory_bounded_buffer() -> None:
    mem = SessionMemory(max_turns=6)
    for i in range(10):
        await mem.add_turn("s1", f"q{i}", f"a{i}")
    history = await mem.get_history("s1")
    assert len(history) == 12
    assert history[0].content == "q4"
    assert history[-1].content == "a9"


@pytest.mark.asyncio
async def test_session_memory_isolation() -> None:
    mem = SessionMemory(max_turns=6)
    await mem.add_turn("s1", "q1", "a1")
    await mem.add_turn("s2", "q2", "a2")
    assert len(await mem.get_history("s1")) == 2
    assert len(await mem.get_history("s2")) == 2
