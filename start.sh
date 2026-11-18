#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  DAG Brain - self-hosted launcher"
echo "============================================================================"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example"
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "[SETUP] Creating the virtual environment..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv
    else
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
    fi
fi

# Install or update dependencies whenever requirements.txt asks for more than .venv has.
if ! .venv/bin/python -m backend.check_env >/dev/null 2>&1; then
    echo "[SETUP] Installing dependencies..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python .venv/bin/python -r requirements.txt
    else
        .venv/bin/pip install -r requirements.txt
    fi
fi

if ! .venv/bin/python -m backend.check_env; then
    echo "[ERROR] Dependencies could not be installed. See the messages above."
    exit 1
fi

echo "[START] Launching DAG Brain on http://127.0.0.1:8000 ..."
.venv/bin/python -m backend.main --open
