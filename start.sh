#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  DAG_Brain (Causal Studio) - One-Click Self-Hosted Launcher"
echo "============================================================================"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example"
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "[SETUP] Setting up Python virtual environment..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv
        uv pip install -r requirements.txt
    else
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
    fi
fi

echo "[START] Launching DAG_Brain Studio on http://127.0.0.1:8000 ..."
.venv/bin/python -m backend.main --open
