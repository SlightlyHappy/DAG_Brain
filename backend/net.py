# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""
Single factory for outbound HTTP clients.

With OFFLINE_MODE enabled (environment variable or runtime setting), every
request to a host other than this machine is refused before it leaves the
process. Local model servers such as Ollama on localhost keep working.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal", "ollama"}


class OfflineBlocked(RuntimeError):
    """Raised when local-only mode stops an outbound request."""


def offline_mode() -> bool:
    if os.getenv("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from .config import get_runtime_settings

        return bool(get_runtime_settings().get("offline_mode"))
    except Exception:
        return False


def is_local(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    return host in LOCAL_HOSTS or host.endswith(".localhost")


async def _guard(request: httpx.Request) -> None:
    if offline_mode() and not is_local(request.url.host):
        raise OfflineBlocked(f"Local-only mode blocked an outbound request to {request.url.host}")


def client(**kwargs: Any) -> httpx.AsyncClient:
    hooks = kwargs.pop("event_hooks", {}) or {}
    hooks.setdefault("request", []).append(_guard)
    return httpx.AsyncClient(event_hooks=hooks, **kwargs)
