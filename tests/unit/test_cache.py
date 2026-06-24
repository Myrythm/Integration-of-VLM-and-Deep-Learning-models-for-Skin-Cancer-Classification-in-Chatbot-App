import json
from pathlib import Path

from services.rag.cache import EmbeddingCache


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(cache_dir=tmp_path / "cache")
    cache.set("hello", [0.1, 0.2, 0.3])
    assert cache.get("hello") == [0.1, 0.2, 0.3]


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(cache_dir=tmp_path / "cache")
    assert cache.get("missing") is None


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    cache1 = EmbeddingCache(cache_dir=tmp_path / "cache")
    cache1.set("hello", [0.1, 0.2])
    cache2 = EmbeddingCache(cache_dir=tmp_path / "cache")
    assert cache2.get("hello") == [0.1, 0.2]
