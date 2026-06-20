from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings
from utils.rag.llm_provider import OpenAIProvider, get_llm_provider


def test_openai_provider_returns_chat_model() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini")
    provider = OpenAIProvider(settings)
    model = provider.get_chat_model()
    assert isinstance(model, BaseChatModel)


def test_openai_provider_streaming() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini")
    provider = OpenAIProvider(settings)
    model = provider.get_streaming_chat_model()
    assert isinstance(model, BaseChatModel)


def test_factory_returns_openai_by_default() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini", llm_backend="openai")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAIProvider)
