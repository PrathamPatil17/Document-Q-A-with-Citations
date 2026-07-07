#!/usr/bin/env bash
# Railway startup script for the FastAPI backend.
# 1. Ingests source documents into ChromaDB (rebuilds from scratch on every boot
#    because Railway containers are ephemeral — no persistent volume on free plan).
# 2. Starts uvicorn on the PORT that Railway injects.

set -euo pipefail

echo "=== Ingesting documents into ChromaDB ==="
python -m src.docqa.ingest

echo "=== Starting FastAPI backend ==="
exec uvicorn src.docqa.api:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
