"""Sweep through similarity thresholds and report recall@5 at each.
Usage: python -m tests.eval.threshold_sweep
"""
import json
from pathlib import Path

from config import get_settings
from utils.rag.app_state import initialize_app_state
from utils.rag.embedder import get_embedder
from utils.rag.vector_store import get_vector_store


def main() -> None:
    settings = get_settings()
    initialize_app_state()

    embedder = get_embedder(settings)
    vector_store = get_vector_store(settings)
    gold = json.loads(Path("tests/eval/gold_set.json").read_text(encoding="utf-8"))

    thresholds = [0.5, 0.6, 0.7, 0.8]
    results_by_threshold: dict[float, dict] = {}

    for th in thresholds:
        hits = 0
        for item in gold:
            emb = embedder.embed_query(item["question"])
            chunks = vector_store.similarity_search(emb, k=5)
            filtered = [c for c in chunks if c["score"] >= th]
            if not filtered:
                continue
            top1 = filtered[0]["metadata"].get("source")
            expected = item["expected_tier"]
            if expected == "guidelines" and top1 in {"aad", "medlineplus", "dermnet"}:
                hits += 1
            elif expected == "pubmed" and top1 == "pubmed":
                hits += 1
        recall = hits / len(gold)
        results_by_threshold[th] = {"recall@5": round(recall, 3), "hits": hits, "total": len(gold)}
        print(f"  threshold={th}: recall@5={recall:.1%} ({hits}/{len(gold)})")

    out = Path("tests/eval/threshold_sweep_results.json")
    out.write_text(json.dumps(results_by_threshold, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")

    best = max(results_by_threshold.items(), key=lambda x: x[1]["recall@5"])
    print(f"\nBest threshold: {best[0]} (recall@5={best[1]['recall@5']:.1%})")
    print(f"Update .env: RAG_SIMILARITY_THRESHOLD={best[0]}")


if __name__ == "__main__":
    main()
