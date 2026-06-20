import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from utils.rag.chain import build_rag_chain, format_docs
from utils.rag.memory import SessionMemory
from utils.rag.prompt import build_prompt_template


def test_format_docs_numbers_chunks() -> None:
    docs = [
        Document(page_content="text one", metadata={"title": "A"}),
        Document(page_content="text two", metadata={"title": "B"}),
    ]
    out = format_docs(docs)
    assert "[1] text one" in out
    assert "[2] text two" in out


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
    history = mem.get_history("s1")
    assert len(history) == 12
    assert history[0].content == "q4"
    assert history[-1].content == "a9"


@pytest.mark.asyncio
async def test_session_memory_isolation() -> None:
    mem = SessionMemory(max_turns=6)
    await mem.add_turn("s1", "q1", "a1")
    await mem.add_turn("s2", "q2", "a2")
    assert len(mem.get_history("s1")) == 2
    assert len(mem.get_history("s2")) == 2
