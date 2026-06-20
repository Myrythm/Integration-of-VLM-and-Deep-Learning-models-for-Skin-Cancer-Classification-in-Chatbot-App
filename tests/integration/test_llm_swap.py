import pytest

from config import Settings
from utils.rag.llm_provider import get_llm_provider


def test_ollama_backend_routes_to_ollama_provider() -> None:
    settings = Settings(
        openai_api_key="test",
        llm_backend="ollama",
        ollama_model="llama3.1:8b",
        ollama_base_url="http://localhost:11434",
    )
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.ollama_base_url == "http://localhost:11434"
    with pytest.raises(NotImplementedError, match="OllamaProvider"):
        get_llm_provider(settings)


def test_vllm_backend_routes_to_vllm_provider() -> None:
    settings = Settings(openai_api_key="test", llm_backend="vllm")
    with pytest.raises(NotImplementedError, match="VLLMProvider"):
        get_llm_provider(settings)


def test_unknown_backend_raises() -> None:
    settings = Settings(openai_api_key="test", llm_backend="anthropic")
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_llm_provider(settings)
