from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_backend: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store_backend: str = "chroma"
    chroma_path: str = "./data/chroma_db"
    chroma_collection: str = "skin_rag_v1"

    rag_similarity_threshold: float = 0.7
    rag_top_k: int = 5
    rag_retrieve_k: int = 10

    session_secret_key: str = "dev-secret-change-me"

    host: str = "0.0.0.0"
    port: int = 8000


def get_settings() -> Settings:
    return Settings()
