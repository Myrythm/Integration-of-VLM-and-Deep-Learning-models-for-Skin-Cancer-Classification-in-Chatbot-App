from pathlib import Path

from config import Settings, get_settings


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_settings_has_tuning_defaults() -> None:
    s = Settings(openai_api_key="x")
    assert s.memory_max_turns == 6
    assert s.llm_temperature == 0.3
    assert s.rag_context_chunk_chars == 1500
    assert s.ingest_batch_size == 100
    assert s.chat_stream_timeout_seconds == 120


def test_runtime_paths_resolved_absolute() -> None:
    s = Settings(openai_api_key="x")
    assert Path(s.chroma_path).is_absolute()
    assert Path(s.model_path).is_absolute()
    assert Path(s.log_dir).is_absolute()


def test_absolute_path_is_left_as_is(tmp_path: Path) -> None:
    target = tmp_path / "custom_chroma"
    s = Settings(openai_api_key="x", chroma_path=str(target))
    assert Path(s.chroma_path) == target
