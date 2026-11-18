# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""
Optional sign-in and hardening for shared deployments.

Local use needs none of this. Set these environment variables to share an
instance with other people:

  DAGBRAIN_PASSWORD   one shared passphrase; when set, every page and API call
                      requires a signed session cookie
  DAGBRAIN_SECRET     key that signs session cookies (random per start if unset,
                      which signs everyone out on restart)
  DAGBRAIN_HOSTED     "1" locks the settings screen: keys, endpoints and privacy
                      options can then only be set by the server's environment
  DAGBRAIN_ORIGINS    comma-separated origins allowed to call the API from other
                      sites; empty (the default) means same-origin only
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
from typing import Dict, List, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

COOKIE = "dagbrain_session"
SESSION_SECONDS = 24 * 3600
MAX_BODY_BYTES = 64 * 1024 * 1024
PUBLIC_PATHS = {"/login", "/api/login", "/api/logout", "/api/health"}

_SECRET = (os.getenv("DAGBRAIN_SECRET") or secrets.token_hex(32)).encode()


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def hosted() -> bool:
    return _truthy(os.getenv("DAGBRAIN_HOSTED"))


def _password() -> str:
    return os.getenv("DAGBRAIN_PASSWORD", "")


def auth_enabled() -> bool:
    return bool(_password())


def allowed_origins() -> List[str]:
    return [o.strip() for o in os.getenv("DAGBRAIN_ORIGINS", "").split(",") if o.strip()]


# ---------------------------------------------------------------- sessions
def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token(now: float | None = None) -> str:
    exp = int((time.time() if now is None else now) + SESSION_SECONDS)
    return f"{exp}.{_sign(str(exp))}"


def valid_token(token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    if not exp.isdigit() or not hmac.compare_digest(sig, _sign(exp)):
        return False
    return int(exp) > (time.time() if now is None else now)


def password_ok(candidate: str) -> bool:
    # Hash both sides so the comparison time does not depend on the length.
    a = hashlib.sha256(candidate.encode()).digest()
    b = hashlib.sha256(_password().encode()).digest()
    return bool(_password()) and hmac.compare_digest(a, b)


# ---------------------------------------------------------------- rate limiting
class Limiter:
    """Per-address lockout plus a global brake on sign-in attempts."""

    def __init__(self, per_ip: int = 5, lock_seconds: int = 900, global_max: int = 50, global_window: int = 600):
        self.per_ip, self.lock_seconds = per_ip, lock_seconds
        self.global_max, self.global_window = global_max, global_window
        self.fails: Dict[str, List[float]] = {}
        self.global_fails: List[float] = []

    def _trim(self, now: float) -> None:
        self.global_fails = [t for t in self.global_fails if now - t < self.global_window]

    def retry_after(self, ip: str, now: float | None = None) -> int:
        now = time.time() if now is None else now
        self._trim(now)
        recent = [t for t in self.fails.get(ip, []) if now - t < self.lock_seconds]
        self.fails[ip] = recent
        wait = 0
        if len(recent) >= self.per_ip:
            wait = int(self.lock_seconds - (now - recent[0])) + 1
        if len(self.global_fails) >= self.global_max:
            wait = max(wait, int(self.global_window - (now - self.global_fails[0])) + 1)
        return wait

    def failed(self, ip: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.fails.setdefault(ip, []).append(now)
        self.global_fails.append(now)

    def succeeded(self, ip: str) -> None:
        self.fails.pop(ip, None)


LIMITER = Limiter()


def client_ip(request: Request) -> str:
    # Behind Cloudflare the real address arrives in CF-Connecting-IP.
    return (
        request.headers.get("cf-connecting-ip")
        or (request.headers.get("x-forwarded-for", "").split(",")[0].strip())
        or (request.client.host if request.client else "unknown")
    )


def _https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"


# ---------------------------------------------------------------- endpoints
async def api_login(request: Request) -> Response:
    if not auth_enabled():
        return JSONResponse({"ok": True, "auth": False})
    ip = client_ip(request)
    wait = LIMITER.retry_after(ip)
    if wait:
        return JSONResponse({"detail": f"Too many attempts. Try again in {max(1, wait // 60)} min."}, status_code=429, headers={"Retry-After": str(wait)})
    try:
        body = await request.json()
    except Exception:
        body = {}
    if password_ok(str(body.get("password", ""))[:200]):
        LIMITER.succeeded(ip)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, make_token(), max_age=SESSION_SECONDS, httponly=True, samesite="lax", secure=_https(request), path="/")
        return resp
    LIMITER.failed(ip)
    await asyncio.sleep(0.6)  # slows guessing without blocking other requests
    return JSONResponse({"detail": "That passphrase is not right."}, status_code=401)


async def api_logout(request: Request) -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><meta name="robots" content="noindex">
<title>Sign in to DAG Brain</title>
<style>
:root{--bg:#f3f4f7;--surface:#fcfcfd;--line:#d9dce4;--text:#1c2030;--text2:#566;--accent:#3a55c4;--bad:#b3261e;--ink:#232838;--on:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#14161d;--surface:#1b1e27;--line:#323747;--text:#eceef4;--text2:#a3a9bb;--accent:#8fa4ff;--bad:#ff8a80;--ink:#e6e9f2;--on:#14161d}}
*{box-sizing:border-box}body{margin:0;min-height:100dvh;display:grid;place-items:center;background:var(--bg);color:var(--text);font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
main{width:min(380px,calc(100vw - 32px));background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:28px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:17px;margin-bottom:4px}
p{margin:0 0 18px;color:var(--text2)}label{display:block;font-weight:500;margin-bottom:6px}
input{width:100%;height:38px;padding:0 12px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit;font:inherit}
input:focus{outline:2px solid var(--accent);outline-offset:1px}
button{margin-top:14px;width:100%;height:38px;border:0;border-radius:6px;background:var(--ink);color:var(--on);font:inherit;font-weight:500;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
#err{min-height:20px;margin-top:10px;color:var(--bad);font-size:13px}
small{display:block;margin-top:18px;color:var(--text2);font-size:12px}small a{color:var(--accent)}
</style></head><body>
<main>
<div class="brand"><svg width="24" height="24" viewBox="0 0 28 28" aria-hidden="true"><path d="M8.5 8.6 18.4 12.9M8.5 19.4l9.9-4.3" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" opacity=".6"/><circle cx="6.5" cy="7.5" r="3.2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="6.5" cy="20.5" r="3.2" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="21.5" cy="14" r="3.6" fill="var(--accent)"/></svg>DAG Brain</div>
<p>This instance is shared by invitation. Enter the passphrase you were given.</p>
<form id="f"><label for="pw">Passphrase</label>
<input id="pw" type="password" autocomplete="current-password" required autofocus>
<button id="go" type="submit">Sign in</button><div id="err" role="alert" aria-live="polite"></div></form>
<small>DAG Brain, created by <a href="https://github.com/SlightlyHappy" rel="noopener noreferrer">SlightlyHappy</a>. <a href="https://github.com/SlightlyHappy/DAG_Brain" rel="noopener noreferrer">Source code</a> (AGPL-3.0).</small>
</main>
<script>
const f=document.getElementById("f"),err=document.getElementById("err"),go=document.getElementById("go");
f.addEventListener("submit",async e=>{e.preventDefault();err.textContent="";go.disabled=true;
try{const r=await fetch("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:document.getElementById("pw").value})});
if(r.ok){location.replace("/");return}
const j=await r.json().catch(()=>({}));err.textContent=j.detail||"Sign-in failed.";}catch(_){err.textContent="Could not reach the server."}
go.disabled=false;});
</script></body></html>"""


async def login_page(request: Request) -> Response:
    if not auth_enabled() or valid_token(request.cookies.get(COOKIE)):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(LOGIN_HTML)


# ---------------------------------------------------------------- middleware
class AuthGate(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Uploads are limited to 64 MB."}, status_code=413)
        if auth_enabled() and request.url.path not in PUBLIC_PATHS and not valid_token(request.cookies.get(COOKIE)):
            if request.url.path.startswith("/api/") or request.method != "GET":
                return JSONResponse({"detail": "Sign in required."}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if request.url.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp
