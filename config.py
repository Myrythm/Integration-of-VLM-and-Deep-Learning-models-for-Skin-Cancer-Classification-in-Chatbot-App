from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_backend: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    vector_store_backend: str = "chroma"
    chroma_path: str = "./data/chroma_db"
    chroma_collection: str = "skin_rag_v1"

    rag_similarity_threshold: float = 0.3
    rag_top_k: int = 5
    rag_retrieve_k: int = 10
    rag_context_chunk_chars: int = 1500

    memory_max_turns: int = 6
    session_memory_max_sessions: int = 1000
    session_memory_ttl_seconds: int = 3600
    llm_temperature: float = 0.3
    ingest_batch_size: int = 100

    chat_stream_timeout_seconds: int = 120

    session_secret_key: str = "dev-secret-change-me"

    model_path: str = "./model/skinCancer.h5"
    image_input_size: int = 224

    log_dir: str = "./logs"

    host: str = "0.0.0.0"
    port: int = 8000

    @field_validator("chroma_path", "model_path", "log_dir")
    @classmethod
    def _anchor_to_project_root(cls, value: str) -> str:
        # Make runtime paths absolute so the app works regardless of the CWD it is
        # launched from. Absolute paths (e.g. test tmp dirs) are left untouched.
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())


@lru_cache
def get_settings() -> Settings:
    return Settings()
