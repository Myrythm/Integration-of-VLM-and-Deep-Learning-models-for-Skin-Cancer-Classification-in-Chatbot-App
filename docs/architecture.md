# Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Web App                                             │
│                                                             │
│  routes/                                                    │
│  ├── api_routes.py      POST /api/upload (image pipeline)   │
│  ├── chat_routes.py     POST /api/chat (SSE streaming)      │
│  └── ui_routes.py       GET / , /upload , /chat (Jinja2)    │
│                                                             │
│  schemas/         Pydantic request/response models          │
│  templates/       Jinja2 HTML (home, upload, chat)          │
│  static/          CSS, JS, EventSource handler              │
│                                                             │
│  services/image/   Image classification                    │
│  └── classifier.py      EfficientNetB3 (.h5) loader + infer │
│                                                             │
│  services/rag/     Framework-agnostic RAG logic             │
│  ├── llm_provider.py    OpenAI / Ollama (stub) / vLLM       │
│  ├── embedder.py        OpenAI (sync + async) / BGE-m3      │
│  ├── vector_store.py    Chroma / Pinecone (stub)            │
│  ├── retriever.py       Evidence-filtered + Chroma adapter  │
│  ├── prompt.py          Single multilingual template        │
│  ├── chain.py           LCEL factory (async retrieve)       │
│  ├── memory.py          Bounded session memory (TTL/LRU)    │
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

- `routes/` may import from `schemas/`, `services/`, FastAPI/Starlette.
- `schemas/` is Pydantic only.
- `services/rag/` modules do NOT import from FastAPI, Starlette, or `routes/`.
- `tests/` may import from anywhere.
