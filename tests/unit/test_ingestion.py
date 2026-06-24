import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config import Settings
from services.rag.ingestion import parse_markdown_file, ingest_directory


def test_parse_markdown_file_extracts_frontmatter(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text(
        """---
source: aad
url: https://aad.org/x
title: Test Title
publish_date: 2024-01-01
language: en
---
# Heading
Body content here.
""",
        encoding="utf-8",
    )
    chunks = parse_markdown_file(md)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "aad"
    assert chunks[0]["metadata"]["url"] == "https://aad.org/x"
    assert chunks[0]["metadata"]["title"] == "Test Title"
    assert "Body content" in chunks[0]["text"]
