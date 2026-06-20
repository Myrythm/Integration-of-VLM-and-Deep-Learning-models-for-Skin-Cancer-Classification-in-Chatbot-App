"""Fetch PubMed abstracts and save as JSON for ingestion.
Run: python data/knowledge_base/pubmed/fetch_pubmed.py --limit 500
"""
import argparse
import json
from pathlib import Path

from utils.rag.pubmed import fetch_pubmed_abstracts

OUT_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.query:
        abstracts = fetch_pubmed_abstracts(query=args.query, max_results=args.limit)
    else:
        abstracts = fetch_pubmed_abstracts(max_results=args.limit)

    for ab in abstracts:
        out_path = OUT_DIR / f"pmid_{ab['pmid']}.json"
        out_path.write_text(json.dumps(ab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(abstracts)} abstracts to {OUT_DIR}")


if __name__ == "__main__":
    main()
