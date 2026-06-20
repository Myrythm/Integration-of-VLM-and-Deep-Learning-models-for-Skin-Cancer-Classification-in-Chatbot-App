import pytest


@pytest.fixture
def settings():
    from config import Settings
    return Settings(
        openai_api_key="test-key",
        session_secret_key="test-secret",
    )
