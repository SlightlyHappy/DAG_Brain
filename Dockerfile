# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATABASE_PATH=/app/data/dag_brain.sqlite

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY LICENSE NOTICE TRADEMARKS.md README.md ARCHITECTURE.md ./
COPY backend ./backend
COPY frontend ./frontend

# Run as an unprivileged user (uid 1000, so a bind-mounted ./data owned by the
# usual first Linux user is writable).
RUN useradd --uid 1000 --create-home dagbrain && mkdir -p /app/data && chown -R 1000:1000 /app/data
USER 1000:1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
