# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""Local-only mode refuses outbound requests before they leave the process."""

import asyncio

import pytest

from backend import net


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _get(url):
    async with net.client(timeout=2.0) as c:
        return await c.get(url)


def test_external_host_blocked(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    with pytest.raises(net.OfflineBlocked):
        _run(_get("https://api.openalex.org/works"))


def test_local_hosts_allowed_by_guard(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    assert net.is_local("localhost") and net.is_local("127.0.0.1") and net.is_local("[::1]")
    assert not net.is_local("api.openai.com")


def test_literature_search_returns_nothing_offline(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    from backend.literature_service import search_scholarly_literature

    assert _run(search_scholarly_literature("sleep cognition", limit=3, engine="all")) == []
