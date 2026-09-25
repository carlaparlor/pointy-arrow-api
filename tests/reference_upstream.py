"""A real, local HTTP implementation of the three upstream protocols this project talks to.

Why this exists
---------------
The sandbox this project is developed in has an egress allowlist that blocks
``gratisfy.xyz``, ``auth.gratisfy.xyz`` and ``api.mail.tm`` (verified: TLS reset
on connect, while github.com / pypi.org are reachable).  Nothing in this file
mocks, stubs or monkeypatches the code under test -- it is a *separate process
speaking real HTTP over a real TCP socket*, so ``pa_server`` / ``pa_router`` /
``pa_credentials`` run completely unmodified against it.

Protocols implemented (reverse-engineered from the client code in this repo):

* ``POST {PA_CHAT_ENDPOINT}``      -- gratisfy.xyz website chat (SSE)
* ``POST {PA_SUPABASE_URL}/...``   -- auth.gratisfy.xyz (Supabase GoTrue)
* ``GET  {PA_MAILTM_BASE}/...``    -- api.mail.tm (Hydra/JSON-LD shapes)

Scenario control
----------------
The behaviour of every request is driven by a JSON *scenario file* whose path is
handed to the process via ``PA_TEST_SCENARIO``.  The file is re-read on every
request, so a test can rewrite it between requests without restarting anything.

Scenario file shape::

    {
      "chat": [                       # consumed FIFO, one entry per POST /api/chat
        {"status": 429, "body": "slow down"},
        {"events": [{"delta": {"content": "hi"}}], "finish": "stop"}
      ],
      "auth": [                       # consumed FIFO by the GoTrue endpoints
        {"status": 400, "body": "{\"error\":\"invalid refresh token\"}"},
        {"access_token": "tok-2", "refresh_token": "ref-2", "expires_in": 3600}
      ],
      "mail": {"domain": "mail.test", "verify_path": "/auth/v1/verify"}
    }

Every request (path, method, headers, body) is appended to an in-memory log that
tests read back over HTTP from ``GET /__log``, which is how the suite asserts on
what the client *actually* sent upstream.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

def _scenario_path() -> str:
    """Read the scenario path on every request.

    The module is imported once per process, but several reference upstreams
    (session-wide and per-test) may each own their own scenario file, so the
    path has to be resolved lazily rather than captured at import time.
    """
    return os.getenv("PA_TEST_SCENARIO", "")

app = FastAPI(title="reference-upstream")

_lock = threading.Lock()
_log: List[Dict[str, Any]] = []
_cursors: Dict[str, int] = {}
_mail_accounts: Dict[str, Dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_scenario() -> Dict[str, Any]:
    path = _scenario_path()
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _set_last_status(status: int) -> None:
    with _lock:
        if _log:
            _log[-1]["response_status"] = status


def _record(request: Request, body: Any) -> None:
    with _lock:
        _log.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
                "ts": time.time(),
            }
        )


def _next(key: str) -> Optional[Dict[str, Any]]:
    """Pop the next unconsumed entry for `key` (one cursor per key).

    Once the list is exhausted the *last* entry sticks, so a scripted failure
    keeps firing for as long as the client retries -- which is exactly what the
    retry/backoff tests need.
    """
    entries = _load_scenario().get(key) or []
    if not entries:
        return None
    with _lock:
        idx = _cursors.get(key, 0)
        _cursors[key] = idx + 1
    return entries[min(idx, len(entries) - 1)]


@app.get("/__scenario")
async def get_scenario() -> Dict[str, Any]:
    return {"path": _scenario_path(), "raw": _load_scenario(), "cursors": dict(_cursors)}


@app.get("/__log")
async def get_log() -> Dict[str, Any]:
    with _lock:
        return {"count": len(_log), "entries": list(_log)}


@app.post("/__reset")
async def reset() -> Dict[str, Any]:
    global _cursors
    with _lock:
        _log.clear()
        _cursors = {}
        _mail_accounts.clear()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# gratisfy.xyz website chat  ->  POST /api/chat  (SSE)
# --------------------------------------------------------------------------- #
def _authorised(request: Request) -> bool:
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        return False
    token = auth[len("Bearer "):].strip()
    # the reference upstream accepts any non-empty token that is not literally
    # "expired-token"; tests use that sentinel to exercise 401 handling.
    return bool(token) and token != "expired-token"


def _wire(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a scenario event into the exact SSE payload the real server sends.

    Scenario events are normally written in the compact form::

        {"delta": {"content": "hi"}}
        {"delta": {"reasoning_content": "..."}}
        {"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                   "function": {"name": "f", "arguments": "{}"}}]}}
        {"finish_reason": "stop"}
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        {"error": {"message": "model not found", "code": "404"}}

    A payload that already carries ``choices`` is passed through verbatim, so the
    suite can also replay byte-exact frames captured from the real upstream.
    """
    if "choices" in ev:
        return ev
    out: Dict[str, Any] = {}
    if "delta" in ev or "finish_reason" in ev:
        choice: Dict[str, Any] = {}
        if "delta" in ev:
            choice["delta"] = ev["delta"]
        if "finish_reason" in ev:
            choice["finish_reason"] = ev["finish_reason"]
        out["choices"] = [choice]
    for key in ("usage", "error", "id", "model", "object"):
        if key in ev:
            out[key] = ev[key]
    return out


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    _record(request, body)

    # Every request consumes exactly one scenario entry, *before* any other
    # decision, so the Nth POST always maps to the Nth scripted step.
    step = _next("chat") or {}

    def _fail(status: int, payload: Any) -> Response:
        _set_last_status(status)
        if isinstance(payload, (dict, list)):
            return JSONResponse(payload, status_code=status)
        return PlainTextResponse(str(payload), status_code=status)

    status = int(step.get("status") or 0)
    if status:
        return _fail(status, step.get("body") or f"upstream {status}")

    if not _authorised(request):
        return _fail(401, {"error": {"message": "website_auth_required",
                                     "code": "website_auth_required"}})

    delay_ms = float(step.get("delay_ms") or 0)
    hang_ms = float(step.get("hang_ms") or 0)

    async def generate():
        if hang_ms:
            time.sleep(hang_ms / 1000.0)
        events = list(step.get("events") or [])
        if step.get("usage") and not any("usage" in e for e in events):
            events.append({"usage": step["usage"]})
        for ev in events:
            nap = float(ev.pop("sleep_ms", 0) or 0)
            if nap:
                time.sleep(nap / 1000.0)
            elif delay_ms:
                time.sleep(delay_ms / 1000.0)
            yield f"data: {json.dumps(_wire(ev), ensure_ascii=False)}\n\n"
        if not step.get("suppress_done"):
            yield "data: [DONE]\n\n"

    _set_last_status(200)
    return StreamingResponse(generate(), media_type="text/event-stream")


def _wire(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a scenario event into the exact SSE payload the real server sends.

    Scenario events are normally written in the compact form::

        {"delta": {"content": "hi"}}
        {"delta": {"reasoning_content": "..."}}
        {"delta": {"tool_calls": [{"index": 0, "id": "call_1",
                                   "function": {"name": "f", "arguments": "{}"}}]}}
        {"finish_reason": "stop"}
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        {"error": {"message": "model not found", "code": "404"}}

    A payload that already carries ``choices`` is passed through verbatim, so the
    suite can also replay byte-exact frames captured from the real upstream.
    """
    if "choices" in ev:
        return ev
    out: Dict[str, Any] = {}
    if "delta" in ev or "finish_reason" in ev:
        choice: Dict[str, Any] = {}
        if "delta" in ev:
            choice["delta"] = ev["delta"]
        if "finish_reason" in ev:
            choice["finish_reason"] = ev["finish_reason"]
        out["choices"] = [choice]
    for key in ("usage", "error", "id", "model", "object"):
        if key in ev:
            out[key] = ev[key]
    return out


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = None
    _record(request, body)

    if not _authorised(request):
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "website_auth_required", "code": "website_auth_required"}},
        )

    step = _next("chat") or {}
    status = int(step.get("status") or 200)
    if status != 200:
        with _lock:
            if _log:
                _log[-1]["response_status"] = status
        return PlainTextResponse(step.get("body") or f"upstream {status}", status_code=status)
    with _lock:
        if _log:
            _log[-1]["response_status"] = 200

# --------------------------------------------------------------------------- #
# auth.gratisfy.xyz  ->  Supabase GoTrue
# --------------------------------------------------------------------------- #
@app.post("/auth/v1/signup")
async def signup(request: Request):
    body = await request.json()
    _record(request, body)
    # signup has its own scenario key: it must not eat the token entries that
    # the subsequent password grant is supposed to consume.
    step = _next("signup") or {}
    if step.get("status"):
        return PlainTextResponse(step.get("body") or "signup failed", status_code=int(step["status"]))
    return JSONResponse(
        {
            "id": "user-ref-1",
            "email": body.get("email"),
            "aud": "authenticated",
            "role": "authenticated",
            "confirmed_at": None,
        }
    )


@app.post("/auth/v1/token")
async def token(request: Request):
    body = await request.json()
    _record(request, body)
    grant = request.query_params.get("grant_type", "")
    step = _next("auth") or {}
    if step.get("status"):
        return PlainTextResponse(step.get("body") or "auth failed", status_code=int(step["status"]))
    suffix = "pw" if grant == "password" else "rt"
    payload = {
        "access_token": step.get("access_token") or f"access-{suffix}",
        "refresh_token": step.get("refresh_token") or f"refresh-{suffix}",
        "expires_in": int(step.get("expires_in") or 3600),
        "token_type": "bearer",
        "user": {"id": "user-ref-1", "email": body.get("email")},
    }
    return JSONResponse(payload)


@app.get("/auth/v1/verify")
async def verify(request: Request):
    _record(request, dict(request.query_params))
    return PlainTextResponse("email confirmed")


# --------------------------------------------------------------------------- #
# api.mail.tm  ->  Hydra/JSON-LD shapes
# --------------------------------------------------------------------------- #
@app.get("/domains")
async def domains(request: Request):
    _record(request, None)
    scenario = _load_scenario()
    mail = scenario.get("mail") or {}
    domain = mail.get("domain") or "mail.test"
    return {
        "hydra:member": [{"id": "dom-1", "domain": domain, "isActive": True, "isPrivate": False}],
        "hydra:totalItems": 1,
    }


@app.post("/accounts")
async def create_account(request: Request):
    body = await request.json()
    _record(request, body)
    with _lock:
        _mail_accounts[body["address"]] = {"password": body["password"], "token": f"mailtok-{len(_mail_accounts)}"}
    return JSONResponse({"id": "acct-1", "address": body["address"]}, status_code=201)


@app.post("/token")
async def mail_token(request: Request):
    body = await request.json()
    _record(request, body)
    with _lock:
        acct = _mail_accounts.get(body.get("address"))
    if not acct or acct["password"] != body.get("password"):
        return PlainTextResponse("bad credentials", status_code=401)
    return {"token": acct["token"]}


@app.get("/messages")
async def messages(request: Request):
    auth = request.headers.get("authorization") or ""
    _record(request, None)
    if not auth.startswith("Bearer mailtok-"):
        return PlainTextResponse("unauthorised", status_code=401)
    scenario = _load_scenario()
    mail = scenario.get("mail") or {}
    verify_path = mail.get("verify_path") or "/auth/v1/verify"
    base = os.getenv("PA_SUPABASE_URL", "http://127.0.0.1")
    text = (
        "Welcome to Gratisfy.\n"
        f"Confirm your address: {base}{verify_path}?token=verify-token-abc&type=signup\n"
    )
    return {
        "hydra:member": [
            {"id": "msg-1", "subject": "Confirm your email", "from": {"address": "noreply@gratisfy.xyz"}}
        ],
        "hydra:totalItems": 1,
    }


@app.get("/messages/{message_id}")
async def read_message(message_id: str, request: Request):
    auth = request.headers.get("authorization") or ""
    _record(request, {"id": message_id})
    if not auth.startswith("Bearer mailtok-"):
        return PlainTextResponse("unauthorised", status_code=401)
    scenario = _load_scenario()
    mail = scenario.get("mail") or {}
    verify_path = mail.get("verify_path") or "/auth/v1/verify"
    base = os.getenv("PA_SUPABASE_URL", "http://127.0.0.1")
    url = f"{base}{verify_path}?token=verify-token-abc&amp;type=signup"
    return {
        "id": message_id,
        "subject": "Confirm your email",
        "intro": "Confirm your email address",
        "text": f"Confirm your address: {base}{verify_path}?token=verify-token-abc&type=signup",
        "html": [f'<a href="{url}">confirm</a>'],
    }
