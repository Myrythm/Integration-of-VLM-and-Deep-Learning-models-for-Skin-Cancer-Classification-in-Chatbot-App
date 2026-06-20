"""Run the eval gold set through the RAG chain and produce a report.

Usage:
    python -m tests.eval.run_eval --output eval_results.json
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document

from config import get_settings
from utils.rag.app_state import initialize_app_state
from utils.rag.citation import extract_citations
from utils.rag.chain import format_docs
from utils.rag.disclaimer import DISCLAIMERS
from utils.rag.embedder import get_embedder
from utils.rag.language import detect_language
from utils.rag.prompt import build_prompt_template
from utils.rag.vector_store import get_vector_store


def evaluate_question(item: dict, chain_components: dict) -> dict:
    question = item["question"]
    language = item["language"]

    query_embedding = chain_components["embedder"].embed_query(question)
    raw_chunks = chain_components["vector_store"].similarity_search(
        query_embedding, chain_components["settings"].rag_retrieve_k
    )
    threshold = chain_components["settings"].rag_similarity_threshold
    filtered = [c for c in raw_chunks if c["score"] >= threshold]
    top_k = filtered[: chain_components["settings"].rag_top_k]

    context_docs = [
        Document(page_content=c["text"], metadata=c["metadata"]) for c in top_k
    ]
    context = format_docs(context_docs)

    response = chain_components["llm"].invoke(
        chain_components["prompt"].format_messages(
            context=context,
            chat_history=[],
            detection="No prior detection.",
            language=language,
            disclaimer=DISCLAIMERS[language],
            question=question,
        )
    )
    response_text = response.content if hasattr(response, "content") else str(response)

    has_disclaimer = DISCLAIMERS[language] in response_text
    citations = extract_citations(response_text)
    response_lower = response_text.lower()
    missing_keywords = [kw for kw in item["must_include"] if kw.lower() not in response_lower]

    expected_source_tier = item["expected_tier"]
    expected_sources = {"guidelines": ["aad", "medlineplus", "dermnet"], "pubmed": ["pubmed"]}[expected_source_tier]
    top1_source = top_k[0]["metadata"].get("source") if top_k else None
    retrieval_hit = top1_source in expected_sources if top_k else False

    return {
        "id": item["id"],
        "question": question,
        "language": language,
        "num_chunks_retrieved": len(raw_chunks),
        "num_chunks_after_filter": len(filtered),
        "top1_source": top1_source,
        "expected_tier": expected_source_tier,
        "retrieval_hit": retrieval_hit,
        "citations_count": len(citations),
        "has_disclaimer": has_disclaimer,
        "missing_keywords": missing_keywords,
        "response_snippet": response_text[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/eval/eval_results.json")
    parser.add_argument("--gold", default="tests/eval/gold_set.json")
    args = parser.parse_args()

    settings = get_settings()
    initialize_app_state()

    from utils.rag.app_state import get_chain
    chain = get_chain()
    prompt = build_prompt_template()
    from utils.rag.llm_provider import get_llm_provider
    llm = get_llm_provider(settings).get_chat_model()
    embedder = get_embedder(settings)
    vector_store = get_vector_store(settings)

    components = {
        "settings": settings,
        "llm": llm,
        "embedder": embedder,
        "vector_store": vector_store,
        "prompt": prompt,
    }

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    results = []
    start = time.time()
    for item in gold:
        try:
            r = evaluate_question(item, components)
        except Exception as e:
            r = {"id": item["id"], "error": str(e)}
        results.append(r)
        print(f"  {item['id']}: {'OK' if r.get('retrieval_hit') and not r.get('missing_keywords') else 'CHECK'}")
    elapsed = time.time() - start

    total = len(results)
    retrieval_recall = sum(1 for r in results if r.get("retrieval_hit")) / total
    disclaimer_rate = sum(1 for r in results if r.get("has_disclaimer")) / total
    keyword_coverage = sum(1 for r in results if not r.get("missing_keywords")) / total

    id_results = [r for r in results if r.get("language") == "id"]
    en_results = [r for r in results if r.get("language") == "en"]
    id_recall = sum(1 for r in id_results if r.get("retrieval_hit")) / max(len(id_results), 1)
    en_recall = sum(1 for r in en_results if r.get("retrieval_hit")) / max(len(en_results), 1)

    summary = {
        "total": total,
        "retrieval_recall@5": round(retrieval_recall, 3),
        "disclaimer_rate": round(disclaimer_rate, 3),
        "keyword_coverage": round(keyword_coverage, 3),
        "id_recall": round(id_recall, 3),
        "en_recall": round(en_recall, 3),
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }

    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults: retrieval_recall@5={retrieval_recall:.1%}, disclaimer={disclaimer_rate:.1%}, keyword_coverage={keyword_coverage:.1%}")
    print(f"  Bilingual parity: ID={id_recall:.1%}, EN={en_recall:.1%}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
