# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI app for an AI skin-cancer education chatbot. Two capabilities: (1) lesion
image classification, and (2) a RAG-grounded bilingual (Indonesian/English) chatbot that
answers from a curated knowledge base with inline citations and a mandatory medical
disclaimer on every turn. It exposes a JSON + SSE API (`/api/upload`, `/api/chat`) **and**
server-rendered Jinja2 pages via `routes/ui_routes.py` (`/`, `/upload`, `/chat`), mounting
`templates/` and `static/`.

## Commands

Python 3.11 only (`requires-python >=3.11,<3.12`). A `.venv` already exists in the repo.

```bash
# Run the dev server (API on :8000, /health, /api/upload, /api/chat)
python -m uvicorn main:app --reload

# All tests
python -m pytest

# Single file / test / by keyword
python -m pytest tests/unit/test_safety.py
python -m pytest tests/unit/test_safety.py::test_name
python -m pytest -k disclaimer

# Ingest knowledge base into Chroma (run before chat works end-to-end)
python -m services.rag.ingestion --source all --rebuild   # NOTE: "all" = aad+medlineplus+dermnet only
python -m services.rag.ingestion --source pubmed          # pubmed must be ingested separately

# Offline RAG quality eval (writes tests/eval/eval_results.json)
python -m tests.eval.run_eval
```

The `scripts/*.sh` helpers (`init_kb.sh`, `run_eval.sh`) are bash and assume `source .venv/bin/activate`;
on this Windows box run them via Git Bash, or just run the underlying `python -m ...` commands above.

## Dependencies

`requirements.txt` is the source of truth (`pip install -r requirements.txt`). `pyproject.toml`
declares an empty `dependencies = []` and exists mainly for `uv` metadata — do not assume it
lists the real deps.

## Configuration

All runtime config flows through `config.py` (`pydantic-settings`, reads `.env`). Copy
`.env.example` → `.env` and set `OPENAI_API_KEY`. Backends and RAG knobs
(`RAG_SIMILARITY_THRESHOLD`, `RAG_TOP_K`, `RAG_RETRIEVE_K`) are all env-driven.

## Architecture

Request flow:
- `POST /api/upload` → `routes/api_routes.py` → `services/image/classifier.classify_skin_image`
  (run via `asyncio.to_thread`) returns a `DetectionResult` + a new `chat_session_id`.
- `POST /api/chat` → `routes/chat_routes.py` streams Server-Sent Events. It runs the safety
  classifier first, then the RAG chain via LangChain `astream_events(version="v2")`, manually
  translating chain events into typed `ChatChunk` SSE frames (`token`, `citation`, `blocked`,
  `done`, `error`). The whole stream is wrapped in `asyncio.timeout`; the retrieve step is
  matched by its `run_name="retrieve"`; citations are emitted after the answer, filtered to the
  `[n]` markers the answer actually cited.

Layering rule (enforced by convention, see `docs/architecture.md`):
- `services/rag/` is framework-agnostic and must **not** import FastAPI, Starlette, or `routes/`.
- `routes/` may import `schemas/`, `services/`, FastAPI.
- `schemas/` is Pydantic only.

Dependency wiring / singletons:
- `services/rag/app_state.py` builds the embedder → vector store → retriever → LLM → LCEL chain
  and `SessionMemory` once, inside the FastAPI `lifespan` (`main.py`), which also warms the Keras
  model. Access the chain/memory only via `get_chain()` / `get_memory()` — these raise
  `RuntimeError` if `initialize_app_state()` hasn't run. Settings come from
  `config.get_settings()` (cached via `lru_cache`). Tests bypass wiring by patching the module
  globals `services.rag.app_state._chain`, `._memory`, `._settings` (see
  `tests/integration/test_chat_route.py`); `memory.get_history` is **async**, so mock it with
  `AsyncMock`. `initialize_app_state()` raises `ValueError` if the OpenAI key is empty.
- The RAG chain (`services/rag/chain.py`) is a pure LCEL pipeline `retrieve | prompt | llm` with
  an **async** `retrieve`; `EvidenceFilteredRetriever` drops chunks below the similarity
  threshold (failing closed when a chunk has no score).

Swappable backends — both use a `Protocol` + factory picking the impl from an env var:
- `llm_provider.py`: `openai` works; `ollama` and `vllm` are deliberate `NotImplementedError` stubs.
- `vector_store.py`: `chroma` works; `pinecone` is a stub.
- `embedder.py`: OpenAI / BGE selectable similarly.
When adding a backend, implement the Protocol and register it in the factory; don't change callers.

Safety & compliance (don't weaken without reason — this is a medical tool):
- `safety.py` heuristically classifies each query; `unsafe_dosage` and `off_topic` queries are
  answered with canned bilingual refusals in `chat_routes.py` and never reach the LLM.
- `disclaimer.py` `force_append_disclaimer()` guarantees the disclaimer is appended to every
  finished response (idempotent). Disclaimers are bilingual and keyed by detected language.
- `language.py` (langdetect) picks `id` vs `en`; the same `prompt.py` template serves both.

## Notable stubs / not-yet-wired

- `services/image/classifier.py` loads the real Keras model from `settings.model_path`
  (`model/skinCancer.h5`), lazily via an `lru_cache` and warmed at startup. It validates the
  output width against `SKIN_CANCER_LABELS` and softmaxes logit-shaped outputs.
- Token accounting **is** wired: `ChatOpenAI(stream_usage=True)` attaches `usage_metadata` to the
  final streamed chunk, which `chat_routes.py` captures and passes to `log_query`.
- `llm_provider.py` exposes a single `get_chat_model()` (no separate streaming variant);
  `ollama`/`vllm` remain `NotImplementedError` stubs, as does `vector_store.PineconeProvider`.

## Tests

`pytest.ini` sets `asyncio_mode=auto`, so `async def test_*` needs no decorator. Layout:
`tests/unit/` (pure module tests), `tests/integration/` (TestClient + mocked chain), and
`tests/eval/` (offline quality harness — `run_eval.py` calls `initialize_app_state()` and hits
the real OpenAI API + a populated Chroma store, so it needs a valid key and prior ingestion;
it is not part of the normal `pytest` run).
