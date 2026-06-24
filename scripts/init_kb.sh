#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and set OPENAI_API_KEY."
    exit 1
fi

python -m services.rag.ingestion --source all --rebuild
echo "Knowledge base initialized."
