import re

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """Extract unique citation numbers from text like 'See [1] and [3].'
    Returns sorted unique list.
    """
    if not text:
        return []
    nums = [int(m.group(1)) for m in _CITATION_RE.finditer(text)]
    return sorted(set(nums))


def format_for_ui(chunks: list[dict], cited_numbers: list[int]) -> list[dict]:
    """Map cited chunk numbers to UI-ready dicts with number, title, url, source."""
    out = []
    for n in cited_numbers:
        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            meta = chunk.get("metadata", {})
            out.append({
                "number": n,
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", ""),
                "source": meta.get("source", "unknown"),
            })
    return out
