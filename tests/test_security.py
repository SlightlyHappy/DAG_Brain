# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""Sign-in gate, hosted-mode lockdown, rate limiting and security headers."""

import pytest
from starlette.testclient import TestClient

from backend import security


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    from backend import database, main

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.sqlite")
    monkeypatch.setenv("DAGBRAIN_PASSWORD", "correct horse battery staple")
    monkeypatch.setenv("DAGBRAIN_HOSTED", "1")
    monkeypatch.setattr(security, "LIMITER", security.Limiter())
    database.init_db()
    with TestClient(main.app, follow_redirects=False) as c:
        yield c


def login(c, pw="correct horse battery staple"):
    return c.post("/api/login", json={"password": pw})


def test_everything_needs_a_session(app_client):
    c = app_client
    assert c.get("/api/projects").status_code == 401
    assert c.post("/api/projects", json={"title": "x"}).status_code == 401
    assert c.get("/static/app.js").status_code == 303 and c.get("/static/app.js").headers["location"] == "/login"
    r = c.get("/")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert c.get("/login").status_code == 200 and "Passphrase" in c.get("/login").text
    assert c.get("/api/health").status_code == 200  # needed by the container health check


def test_health_reveals_no_study_data_and_reports_flags(app_client):
    h = app_client.get("/api/health").json()
    assert h["hosted"] is True and h["auth"] is True and "projects" not in h


def test_wrong_password_rejected_right_one_sets_a_safe_cookie(app_client):
    c = app_client
    assert login(c, "wrong").status_code == 401
    r = login(c)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert c.get("/api/projects").status_code == 200
    assert c.get("/").status_code == 200
    assert c.get("/login").status_code == 303  # already signed in


def test_secure_flag_when_behind_https_proxy(app_client):
    r = app_client.post("/api/login", json={"password": "correct horse battery staple"}, headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_forged_and_expired_tokens_rejected(app_client):
    c = app_client
    c.cookies.set(security.COOKIE, "9999999999.deadbeef")
    assert c.get("/api/projects").status_code == 401
    c.cookies.clear()
    expired = security.make_token(now=1_000_000)
    c.cookies.set(security.COOKIE, expired)
    assert c.get("/api/projects").status_code == 401
    assert security.valid_token(security.make_token())


def test_logout_ends_the_session(app_client):
    c = app_client
    login(c)
    c.post("/api/logout")
    c.cookies.clear()
    assert c.get("/api/projects").status_code == 401


def test_lockout_after_repeated_failures(app_client):
    c = app_client
    for _ in range(5):
        assert login(c, "nope").status_code == 401
    r = login(c)  # even the right passphrase is refused while locked
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_limiter_unit():
    lim = security.Limiter(per_ip=3, lock_seconds=100, global_max=5, global_window=100)
    for _ in range(3):
        lim.failed("a", now=0)
    assert lim.retry_after("a", now=1) > 0 and lim.retry_after("b", now=1) == 0
    assert lim.retry_after("a", now=200) == 0
    for i in range(5):
        lim.failed(f"x{i}", now=300)
    assert lim.retry_after("fresh", now=301) > 0  # global brake


def test_hosted_mode_locks_settings(app_client):
    c = app_client
    login(c)
    r = c.post("/api/settings", json={"openrouter_api_key": "sk-x", "ollama_base_url": "http://169.254.169.254"})
    assert r.status_code == 403
    assert c.get("/api/settings").status_code == 200  # reading (masked) still works


def test_no_cross_site_access_by_default(app_client):
    r = app_client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_security_headers_and_oversize_upload(app_client):
    c = app_client
    login(c)
    r = c.get("/api/projects")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
    big = c.post("/api/projects/proj_sleep_cognition/datasets", headers={"content-length": str(200 * 1024 * 1024)}, content=b"x")
    assert big.status_code == 413


def test_open_when_no_password_is_set(tmp_path, monkeypatch):
    from backend import database, main

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t2.sqlite")
    monkeypatch.delenv("DAGBRAIN_PASSWORD", raising=False)
    monkeypatch.delenv("DAGBRAIN_HOSTED", raising=False)
    database.init_db()
    with TestClient(main.app, follow_redirects=False) as c:
        assert c.get("/api/projects").status_code == 200
        assert c.post("/api/settings", json={}).status_code != 403
