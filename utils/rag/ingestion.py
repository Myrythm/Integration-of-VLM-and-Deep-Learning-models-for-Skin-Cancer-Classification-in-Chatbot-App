import argparse
import hashlib
import json
from pathlib import Path

import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from utils.rag.embedder import get_embedder
from utils.rag.vector_store import get_vector_store

KB_ROOT = Path("data/knowledge_base")
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def parse_pubmed_json(path: Path) -> list[dict]:
    """Parse a PubMed abstract JSON file into chunk dicts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pmid = data.get("pmid", "")
    abstract = data.get("abstract", "")
    if not abstract:
        return []
    text = f"{data.get('title', '')}\n\n{abstract}"
    chunks = SPLITTER.split_text(text)
    out = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = hashlib.sha256(f"pubmed:{pmid}::{i}".encode()).hexdigest()
        out.append({
            "id": f"sha256:{chunk_id}",
            "text": chunk_text,
            "metadata": {
                "source": "pubmed",
                "url": data.get("url", ""),
                "title": data.get("title", ""),
                "publish_date": data.get("publish_date", ""),
                "language": "en",
                "pmid": pmid,
                "journal": data.get("journal", ""),
            },
        })
    return out


def parse_markdown_file(path: Path) -> list[dict]:
    post = frontmatter.load(path)
    meta = dict(post.metadata)
    chunks = SPLITTER.split_text(post.content)
    out = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = hashlib.sha256(
            f"{meta.get('url', path.name)}::{i}::{chunk_text[:80]}".encode()
        ).hexdigest()
        out.append({
            "id": f"sha256:{chunk_id}",
            "text": chunk_text,
            "metadata": {
                "source": meta.get("source", "unknown"),
                "url": meta.get("url", ""),
                "title": meta.get("title", path.stem),
                "publish_date": str(meta.get("publish_date", "")),
                "language": meta.get("language", "en"),
            },
        })
    return out


def ingest_directory(source: str) -> int:
    """Parse all markdown files in data/knowledge_base/{source}/ and upsert to Chroma.

    Returns the number of chunks ingested.
    """
    settings = get_settings()
    src_dir = KB_ROOT / source
    if not src_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {src_dir}")

    all_chunks: list[dict] = []
    for f in sorted(src_dir.iterdir()):
        if f.suffix == ".md":
            all_chunks.extend(parse_markdown_file(f))
        elif f.suffix == ".json" and source == "pubmed":
            all_chunks.extend(parse_pubmed_json(f))

    if not all_chunks:
        print(f"No .md or .json files found in {src_dir}")
        return 0

    from utils.rag.cache import EmbeddingCache
    cache = EmbeddingCache()
    embedder = get_embedder(settings)
    vector_store = get_vector_store(settings)

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        cached = [cache.get(t) for t in texts]
        missing_idx = [j for j, v in enumerate(cached) if v is None]
        missing_texts = [texts[j] for j in missing_idx]
        if missing_texts:
            new_embeddings = embedder.embed_documents(missing_texts)
            for j, emb in zip(missing_idx, new_embeddings):
                cached[j] = emb
                cache.set(texts[j], emb)
        vector_store.upsert(batch, cached)
        print(f"Ingested batch {i // batch_size + 1}: {len(batch)} chunks (cache hits: {len(texts) - len(missing_idx)})")

    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge base into vector store")
    parser.add_argument(
        "--source",
        choices=["aad", "medlineplus", "dermnet", "pubmed", "all"],
        required=True,
    )
    parser.add_argument("--rebuild", action="store_true", help="Wipe collection before ingest")
    args = parser.parse_args()

    if args.rebuild:
        settings = get_settings()
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            client.delete_collection(settings.chroma_collection)
            print(f"Wiped collection: {settings.chroma_collection}")
        except Exception:
            pass

    sources = ["aad", "medlineplus", "dermnet"] if args.source == "all" else [args.source]
    total = 0
    for src in sources:
        n = ingest_directory(src)
        total += n
        print(f"  {src}: {n} chunks")
    print(f"Total chunks ingested: {total}")


if __name__ == "__main__":
    main()
