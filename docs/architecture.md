# Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Web App                                             │
│                                                             │
│  routes/                                                    │
│  ├── api_routes.py      POST /api/upload (image pipeline)   │
│  └── chat_routes.py     POST /api/chat (SSE streaming)      │
│                                                             │
│  schemas/         Pydantic request/response models          │
│  templates/       Jinja2 HTML (home, upload, chat)          │
│  static/          CSS, JS, EventSource handler              │
│                                                             │
│  utils/rag/        Framework-agnostic RAG logic             │
│  ├── llm_provider.py    OpenAI / Ollama (stub) / vLLM       │
│  ├── embedder.py        OpenAI / BGE-m3                     │
│  ├── vector_store.py    Chroma / Pinecone (stub)            │
│  ├── retriever.py       Evidence-filtered retriever         │
│  ├── prompt.py          Single multilingual template        │
│  ├── chain.py           LCEL factory                        │
│  ├── memory.py          Bounded session memory              │
│  ├── safety.py          Query classifier                     │
│  ├── disclaimer.py      Three-layer disclaimer              │
│  ├── language.py        Langdetect wrapper                  │
│  ├── citation.py        Inline citation extractor           │
│  ├── cache.py           Embedding cache                     │
│  ├── logging_config.py  Structured logging                  │
│  ├── ingestion.py       CLI for KB ingestion                │
│  ├── pubmed.py          PubMed E-utilities client           │
│  └── app_state.py       Lifespan-managed singletons         │
└─────────────────────────────────────────────────────────────┘
```

## Boundaries

- `routes/` may import from `schemas/`, `utils/`, FastAPI/Starlette.
- `schemas/` is Pydantic only.
- `utils/rag/` modules do NOT import from FastAPI, Starlette, or `routes/`.
- `tests/` may import from anywhere.
