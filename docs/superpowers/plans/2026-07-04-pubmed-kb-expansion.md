# PubMed Knowledge Base Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the empty `data/knowledge_base/pubmed/` slice with a verified batch of up to 50
PubMed abstracts, ingest them into the existing Chroma collection, and confirm they are actually
retrievable — without touching the aad/dermnet/medlineplus data.

**Architecture:** Two-stage batch pipeline using code that already exists and needs no changes:
`data/knowledge_base/pubmed/fetch_pubmed.py` calls `services/rag/pubmed.py:fetch_pubmed_abstracts`
against `eutils.ncbi.nlm.nih.gov` and writes `pmid_*.json` files; then
`python -m services.rag.ingestion --source pubmed` (no `--rebuild`) chunks, embeds, and upserts
them into the same Chroma collection the aad/dermnet/medlineplus chunks already live in. A final
verification stage runs a direct `ChromaProvider.similarity_search` call and reports the active
`RAG_SIMILARITY_THRESHOLD`.

**Tech Stack:** Python 3.11, existing modules only — `services/rag/pubmed.py`,
`services/rag/ingestion.py`, `services/rag/vector_store.py` (`ChromaProvider`),
`services/rag/embedder.py` (`OpenAIEmbedder`). No new source files. One small source edit is
required (see Task 1, Step 2) — discovered during execution, not anticipated by the original spec.

## Amendment (discovered during execution)

Task 1's first execution attempt found that `services/rag/pubmed.py:DEFAULT_QUERY` returns **0**
PubMed results as written — the combination of `"patient education"[MeSH]` AND
`free full text[Filter]` is too restrictive for what PubMed currently has indexed (confirmed: a
bare `melanoma` query returns results; network/API access is fine). The spec assumed
`DEFAULT_QUERY` would just work; it doesn't. Per user decision, Task 1 now includes relaxing
`DEFAULT_QUERY` by dropping the `"patient education"[MeSH]` clause (keeping the skin-cancer terms,
`English`, and `free full text[Filter]`) as its first step, before fetching.

## Global Constraints

- Do not modify or re-ingest the existing aad/dermnet/medlineplus data or chunks — this task only
  *adds* PubMed content. (spec: Approach)
- Never pass `--rebuild` to `services.rag.ingestion` — it wipes the whole collection.
  (spec: Why not `--rebuild`)
- Cap this run at `--limit 50` abstracts — this is a verification batch, not the full ~500 implied
  by `docs/knowledge-base.md`. (spec: Out of scope)
- Do not change `RAG_SIMILARITY_THRESHOLD` in `.env` or `config.py` — report its effective value
  only; the user decides whether to lower it. (spec: Out of scope, Risks)
- Requires a valid `OPENAI_API_KEY` in `.env` (used for embeddings during ingestion and the
  verification query) and outbound network access to `eutils.ncbi.nlm.nih.gov`. (spec: Risks)
- Chunk IDs are SHA256-based, so re-running fetch/ingest later is idempotent — a failed or partial
  run can be safely retried. (spec: Risks)

---

### Task 1: Fetch PubMed abstracts

**Files:**
- Modify: `services/rag/pubmed.py:9-12` (`DEFAULT_QUERY` — drop the `"patient education"[MeSH]`
  clause only; see Amendment above)
- Uses (no edits): `data/knowledge_base/pubmed/fetch_pubmed.py`
- Creates: `data/knowledge_base/pubmed/pmid_*.json` (up to 50 files)

**Interfaces:**
- Consumes: `fetch_pubmed_abstracts(query: str = DEFAULT_QUERY, max_results: int = 500) -> list[dict]`
  (`services/rag/pubmed.py:68`) via the script's CLI — signature unchanged, only the `DEFAULT_QUERY`
  string constant changes.
- Produces: JSON files shaped `{pmid, title, abstract, journal, publish_date, url}`, one per
  `pmid_*.json`, consumed by Task 2's `ingest_directory("pubmed")`.

- [ ] **Step 1: Relax `DEFAULT_QUERY`**

In `services/rag/pubmed.py`, change:
```python
DEFAULT_QUERY = (
    '("skin neoplasms"[MeSH] OR melanoma OR "basal cell" OR "squamous cell") '
    'AND "patient education"[MeSH] AND English AND free full text[Filter]'
)
```
to:
```python
DEFAULT_QUERY = (
    '("skin neoplasms"[MeSH] OR melanoma OR "basal cell" OR "squamous cell") '
    'AND English AND free full text[Filter]'
)
```
Do not touch anything else in the file. `tests/unit/test_pubmed.py` does not assert on
`DEFAULT_QUERY`'s exact text (confirmed via grep before this amendment), so no test changes are
expected — run `python -m pytest tests/unit/test_pubmed.py -v` after the edit as a quick check
anyway.

- [ ] **Step 2: Record the baseline PubMed file count**

Run:
```bash
python -c "from pathlib import Path; print(len(list(Path('data/knowledge_base/pubmed').glob('pmid_*.json'))))"
```
Expected: `0` (the directory currently holds only `.gitkeep` and `fetch_pubmed.py`). If it's
nonzero, record that number as `BASELINE_FILES` and use it instead of `0` in Step 4's check.

- [ ] **Step 3: Run the fetch script capped at 50 abstracts**

Run: `python data/knowledge_base/pubmed/fetch_pubmed.py --limit 50`

Expected: stdout ending in `Saved N abstracts to <...>\pubmed`, with `1 <= N <= 50`. If `N == 0`
again even with the relaxed query, stop — do not proceed to Task 2, and escalate rather than
relaxing the query further on your own.

- [ ] **Step 4: Verify the files landed on disk**

Run the same command as Step 2:
```bash
python -c "from pathlib import Path; print(len(list(Path('data/knowledge_base/pubmed').glob('pmid_*.json'))))"
```
Expected: `BASELINE_FILES + N` (matches Step 3's reported count).

- [ ] **Step 5: Commit the query fix separately from the fetched data**

```bash
git add services/rag/pubmed.py
git commit -m "fix: relax PubMed DEFAULT_QUERY — patient-education MeSH filter matched zero articles"
```
Expected: commit succeeds and touches only `services/rag/pubmed.py` — the `pmid_*.json` files stay
unstaged for Task 4.

---

### Task 2: Ingest the fetched abstracts into Chroma

**Files:**
- Uses (no edits): `services/rag/ingestion.py`
- Reads: `data/knowledge_base/pubmed/pmid_*.json` (from Task 1)
- Side effect (not a source file): upserts into the Chroma collection at `settings.chroma_path` /
  `settings.chroma_collection` — this directory is `data/chroma_db/`, which is in `.gitignore`, so
  nothing here needs a commit.

**Interfaces:**
- Consumes: `ingest_directory(source: str) -> int` (`services/rag/ingestion.py:72`) via
  `python -m services.rag.ingestion --source pubmed` (no `--rebuild`).
- Produces: a chunk-count delta consumed by Task 3's verification.

- [ ] **Step 1: Record the pre-ingestion chunk count**

Run:
```bash
python -c "from config import get_settings; from services.rag.vector_store import get_vector_store; print(get_vector_store(get_settings())._collection.count())"
```
Expected: prints an integer. Record it as `BEFORE_COUNT` (per the design doc this is currently
~6, but the real number on this machine is whatever the command prints — use that, not the
document's estimate).

- [ ] **Step 2: Run ingestion for the pubmed source only**

Run: `python -m services.rag.ingestion --source pubmed`

Expected: stdout ends with `pubmed: M chunks` then `Total chunks ingested: M`, with `M > 0`. The
`--rebuild` flag must **not** be passed — confirm the command above matches exactly.

- [ ] **Step 3: Record the post-ingestion chunk count and confirm the increase**

Run the same command as Step 1.

Expected: `AFTER_COUNT > BEFORE_COUNT`, and ideally `AFTER_COUNT - BEFORE_COUNT == M` from Step 2.
If `AFTER_COUNT == BEFORE_COUNT` (no increase), check whether this exact batch of PMIDs was
already ingested in a prior run (chunk IDs are SHA256-based and idempotent, so re-ingesting the
same PMIDs upserts 0 *new* chunks) before treating this as a failure.

No commit — this step only mutates the local, gitignored Chroma store.

---

### Task 3: Verify retrieval and report the effective similarity threshold

**Files:**
- Uses (no edits): `services/rag/vector_store.py`, `services/rag/embedder.py`, `config.py`

**Interfaces:**
- Consumes: `ChromaProvider.similarity_search(query_embedding: list[float], k: int) -> list[dict]`
  (`services/rag/vector_store.py:43`), `OpenAIEmbedder.embed_query(text: str) -> list[float]`
  (`services/rag/embedder.py:21`), `Settings.rag_similarity_threshold` (`config.py:24`).
- Produces: a printed report (no files written) — this is what gets relayed back to the user per
  the spec's success criteria.

- [ ] **Step 1: Run a direct similarity search for a covered topic**

Run:
```bash
python -c "
from config import get_settings
from services.rag.embedder import get_embedder
from services.rag.vector_store import get_vector_store

settings = get_settings()
embedder = get_embedder(settings)
vs = get_vector_store(settings)

query_embedding = embedder.embed_query('melanoma treatment')
results = vs.similarity_search(query_embedding, k=5)

for r in results:
    print(r['metadata'].get('source'), round(r['score'], 4), r['metadata'].get('title', '')[:60])
"
```
Expected: at least one printed line with `source == pubmed` and a plausible cosine-similarity
score. Judge by "at least one non-trivial pubmed hit," not an exact score — exact values depend on
which abstracts the query happened to fetch.

- [ ] **Step 2: Report the effective `RAG_SIMILARITY_THRESHOLD`**

Run: `python -c "from config import get_settings; print(get_settings().rag_similarity_threshold)"`

Expected: prints a float. Compare it against the pubmed scores from Step 1: if the printed
threshold is higher than those scores, this means `EvidenceFilteredRetriever` would currently drop
the PubMed content at query time even though it's in the collection — call this out explicitly in
the final report to the user. Per the design doc this is **report only**; do not edit `.env` or
`config.py` to change it.

- [ ] **Step 3: Run the full test suite as a regression check**

Run: `python -m pytest`

Expected: same pass/skip counts as before this plan started. No source files were touched by
Tasks 1–3, so this is a sanity check that the ad hoc commands above didn't disturb the environment
(e.g. leaving behind stray state or a broken import).

No commit — this task only reads and prints.

---

### Task 4: Commit the newly fetched PubMed knowledge base files

**Files:**
- Create (git-tracked): `data/knowledge_base/pubmed/pmid_*.json` (written by Task 1)

**Interfaces:** None — version-control step only, no code.

- [ ] **Step 1: Stage only the new PubMed JSON files**

Run: `git add data/knowledge_base/pubmed/pmid_*.json`

Expected: `git status` shows only the new `pmid_*.json` files staged. Confirm no other files were
swept in — this plan does not touch anything else.

- [ ] **Step 2: Commit**

```bash
git commit -m "data: ingest initial PubMed abstract batch into knowledge base"
```
Expected: commit succeeds; `git log -1 --stat` shows only the `pmid_*.json` files changed.

---

## Final report to the user (after Task 4)

State explicitly, using the values captured above:
- How many abstracts were fetched and ingested (`N` from Task 1, `M` from Task 2).
- Chunk count before/after (`BEFORE_COUNT` / `AFTER_COUNT` from Task 2).
- At least one example PubMed similarity-search hit with its score (Task 3, Step 1).
- The current effective `RAG_SIMILARITY_THRESHOLD` and whether it appears to already filter out
  the newly ingested PubMed content (Task 3, Step 2) — flag this for the user's decision; do not
  change it as part of this plan.
