from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings
from services.rag.llm_provider import OpenAIProvider, get_llm_provider


def test_openai_provider_returns_chat_model() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini")
    provider = OpenAIProvider(settings)
    model = provider.get_chat_model()
    assert isinstance(model, BaseChatModel)


def test_get_chat_model_enables_stream_usage() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini")
    provider = OpenAIProvider(settings)
    model = provider.get_chat_model()
    assert model.stream_usage is True


def test_provider_has_no_separate_streaming_method() -> None:
    provider = OpenAIProvider(Settings(openai_api_key="test-key"))
    assert not hasattr(provider, "get_streaming_chat_model")


def test_get_chat_model_uses_configured_temperature() -> None:
    settings = Settings(openai_api_key="test-key", llm_temperature=0.7)
    model = OpenAIProvider(settings).get_chat_model()
    assert model.temperature == 0.7


def test_factory_returns_openai_by_default() -> None:
    settings = Settings(openai_api_key="test-key", openai_model="gpt-4o-mini", llm_backend="openai")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAIProvider)
