from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda


def format_docs(docs: list[Document]) -> str:
    """Number chunks [1], [2], ... for inline citation. Truncates long content."""
    if not docs:
        return "(no context available)"
    parts = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "")
        source = d.metadata.get("source", "unknown")
        url = d.metadata.get("url", "")
        header = f"(source: {source}"
        if title:
            header += f", title: {title}"
        if url:
            header += f", url: {url}"
        header += ")"
        content = d.page_content[:1500]
        parts.append(f"[{i}] {content}\n{header}")
    return "\n\n".join(parts)


def build_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
) -> Runnable:
    """Build the LCEL RAG chain. Stateless beyond injected deps."""

    def retrieve(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {**inputs, "context": format_docs(docs), "retrieved_docs": docs}

    return (
        RunnableLambda(retrieve)
        | prompt
        | llm
    )
