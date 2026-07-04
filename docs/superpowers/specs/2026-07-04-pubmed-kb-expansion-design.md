# PubMed Knowledge Base Expansion — Design

**Date:** 2026-07-04
**Status:** Approved

## Problem

The RAG knowledge base is nearly empty: the Chroma collection holds only ~6 chunks total,
sourced from one sample markdown file each in `data/knowledge_base/{aad,dermnet,medlineplus}/`.
`data/knowledge_base/pubmed/` contains only the fetch script (`fetch_pubmed.py`) — it has never
been run, so no PubMed abstracts have been ingested.

This was confirmed as the root cause of a production observation: `/api/chat` returned
`num_chunks_retrieved: 0` / `num_chunks_after_filter: 0` for a real question, even though the
request succeeded (200 OK) and the LLM still produced an answer — just ungrounded, with no
citations.

## Goal

Populate the PubMed slice of the knowledge base with a small, verifiable batch of abstracts so
retrieval has real content to return, without touching the existing aad/dermnet/medlineplus
data.

## Approach

Incremental fetch + ingest, no rebuild:

1. Run `python data/knowledge_base/pubmed/fetch_pubmed.py --limit 50` using the module's
   existing `DEFAULT_QUERY` (skin neoplasms / melanoma / BCC / SCC, patient education, English,
   free full text). Writes up to 50 `pmid_*.json` files into `data/knowledge_base/pubmed/`.
2. Run `python -m services.rag.ingestion --source pubmed` (no `--rebuild`) to append the new
   chunks into the existing Chroma collection. The 3 existing aad/dermnet/medlineplus chunks are
   left untouched.
3. Verify:
   - Chroma collection chunk count increased by the expected amount.
   - A direct `ChromaProvider.similarity_search` call for a covered topic (e.g. "melanoma
     treatment") returns PubMed-sourced chunks with plausible scores.
   - Report the current `RAG_SIMILARITY_THRESHOLD` value from the active `.env` (or its
     default) — `.env.example` suggests `0.7` while `config.py`'s default is `0.3`; a `0.7`
     threshold could still zero out results even with data present. This is a **report only**,
     not an automatic change — the user decides whether to lower it.

### Why not `--rebuild`

`--rebuild` wipes the whole collection before re-ingesting. Since this task only *adds* PubMed
content, rebuilding is unnecessary churn and risks re-ingestion mistakes affecting the
already-working aad/dermnet/medlineplus chunks.

## Out of scope

- Researching/writing additional AAD, MedlinePlus, or DermNet content for Basal Cell Carcinoma,
  Squamous Cell Carcinoma, or Nevus (all currently thin or entirely uncovered). Deferred to a
  future task at the user's request.
- Scaling PubMed ingestion beyond 50 abstracts (this is a verification batch, not the full
  ~500 implied by `docs/knowledge-base.md`).
- Changing `RAG_SIMILARITY_THRESHOLD` — reported, not modified.

## Success criteria

- `data/knowledge_base/pubmed/` contains new `pmid_*.json` files (up to 50, fewer if the query
  returns fewer results).
- Chroma collection chunk count is strictly greater than before ingestion.
- A direct similarity search against a skin-cancer topic returns at least one PubMed-sourced
  result.
- The current effective `RAG_SIMILARITY_THRESHOLD` is reported back to the user.

## Risks / notes

- Depends on outbound network access to `eutils.ncbi.nlm.nih.gov` — confirmed reachable from
  this environment before this spec was written.
- Rate limiting (0.4s between batches of 200 PMIDs) is already handled inside
  `services/rag/pubmed.py`; irrelevant at a 50-item scale (single batch).
- Ingestion is idempotent per existing design (chunk IDs are SHA256-based), so re-running this
  spec's steps later is safe.
