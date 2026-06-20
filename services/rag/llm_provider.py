from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import Settings


class LLMProvider(Protocol):
    def get_chat_model(self) -> BaseChatModel: ...
    def get_streaming_chat_model(self) -> BaseChatModel: ...


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_model,
            temperature=0.3,
        )

    def get_streaming_chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_model,
            temperature=0.3,
            streaming=True,
        )


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_backend == "openai":
        return OpenAIProvider(settings)
    if settings.llm_backend == "ollama":
        raise NotImplementedError("OllamaProvider not yet implemented (Phase 3 stub)")
    if settings.llm_backend == "vllm":
        raise NotImplementedError("VLLMProvider not yet implemented (Phase 3 stub)")
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")
