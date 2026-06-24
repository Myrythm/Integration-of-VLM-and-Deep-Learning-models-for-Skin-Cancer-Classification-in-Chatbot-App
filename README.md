# Skin Cancer RAG-Enhanced Chatbot

A FastAPI web app for AI-assisted skin-cancer education. Combines EfficientNetB3 lesion classification with a RAG-grounded bilingual chatbot, designed to give laypeople trustworthy, citation-backed answers about their skin.

## Features

- 🩺 **AI lesion classification** via EfficientNetB3 (4 classes: BCC, SCC, Melanoma, Nevus)
- 💬 **Bilingual chatbot** (Indonesian + English) grounded in patient-education guidelines and curated PubMed abstracts
- 📚 **Source citation** on every response — click to verify
- ⚠️ **Mandatory medical disclaimer** at three layers (system, post-generation, UI)
- 🔒 **Privacy-friendly** — local ChromaDB vector store, queries hashed before logging
- 🔌 **Swappable LLM** — OpenAI today, Ollama/vLLM tomorrow via env var

## Quickstart

```bash
git clone <repo>
cd <repo>
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENAI_API_KEY

# Place the trained model file at ./model/skinCancer.h5
# (see model/README.md for details)

./scripts/init_kb.sh
python -m uvicorn main:app --reload
```

Open http://localhost:8000

## Image Classification Model

The `POST /api/upload` endpoint uses an EfficientNetB3 model trained on skin lesion images. The model:

- Takes 224×224 RGB image input
- Outputs 4-class probabilities: `Karsinoma Sel Basal`, `Karsinoma Sel Skuamosa`, `Melanoma`, `Nevus`
- Returns the predicted class label and confidence score
- Is loaded lazily on first request and cached

**Setup**: Place `skinCancer.h5` (~134 MB) in `./model/` (or set `MODEL_PATH` in `.env` to override). See [model/README.md](model/README.md) for full details.

**Required**: `tensorflow==2.15.0` (already pinned in `requirements.txt`).

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
