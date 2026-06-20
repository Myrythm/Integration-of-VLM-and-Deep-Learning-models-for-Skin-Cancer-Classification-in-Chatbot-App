# Skin Cancer RAG-Enhanced Chatbot

A FastAPI web app for AI-assisted skin-cancer education. Combines EfficientNetB3 lesion classification with a RAG-grounded bilingual chatbot, designed to give laypeople trustworthy, citation-backed answers about their skin.

## Features

- 🩺 **AI lesion classification** via EfficientNetB3
- 💬 **Bilingual chatbot** (Indonesian + English) grounded in patient-education guidelines and curated PubMed abstracts
- 📚 **Source citation** on every response — click to verify
- ⚠️ **Mandatory medical disclaimer** at three layers (system, post-generation, UI)
- 🔒 **Privacy-friendly** — local ChromaDB vector store, queries hashed before logging
- 🔌 **Swappable LLM** — OpenAI today, Ollama/vLLM tomorrow via env var

## Quickstart

```bash
git clone <repo>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENAI_API_KEY
./scripts/init_kb.sh
python -m uvicorn main:app --reload
```

Open http://localhost:8000

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full diagram.

## Knowledge Base

See [docs/knowledge-base.md](docs/knowledge-base.md) for source licensing and ingestion.

## Evaluation

```bash
./scripts/run_eval.sh
```

Outputs `tests/eval/eval_results.json` with retrieval recall@5, disclaimer rate, and bilingual parity.

## Project Structure

See [docs/superpowers/specs/2026-06-20-skin-cancer-rag-design.md](docs/superpowers/specs/2026-06-20-skin-cancer-rag-design.md) for full design spec.

## License

MIT (project code). Knowledge base sources retain their own licenses — see [docs/knowledge-base.md](docs/knowledge-base.md).

## Disclaimer

This is an educational tool, NOT a diagnostic device. Always consult a qualified dermatologist for diagnosis and treatment. The chatbot is built to remind users of this at every turn.
