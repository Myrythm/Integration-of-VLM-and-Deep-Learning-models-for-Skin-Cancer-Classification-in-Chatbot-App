import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """File-based cache for embeddings, keyed by SHA256 of input text."""

    def __init__(self, cache_dir: Path | str = ".cache/embeddings") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def get(self, text: str) -> list[float] | None:
        path = self._path(self._key(text))
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, text: str, embedding: list[float]) -> None:
        path = self._path(self._key(text))
        path.write_text(json.dumps(embedding), encoding="utf-8")
