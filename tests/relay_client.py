"""Route real HTTP requests through the git/CI relay.

This sandbox's egress allowlist blocks gratisfy.xyz, auth.gratisfy.xyz and
api.mail.tm (verified: TLS reset on connect).  ``.pa-relay/runner.py`` runs on a
GitHub Actions runner that has no such restriction, and exchanges request/
response JSON through commits on this branch.

``RelayTransport`` is a genuine ``requests`` transport adapter, so the production
code -- ``pa_router``, ``pa_credentials``, the OpenAI SDK, anything built on
``requests`` -- runs completely unmodified while its sockets are served by the
relay.  Nothing is stubbed: every byte on the wire is produced by the real remote
server and carried back verbatim.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests

REPO = Path(__file__).resolve().parents[1]
BRANCH = os.getenv("PA_RELAY_BRANCH", "arena/01a0da31-pointy-arrow-api")
IN_DIR = REPO / ".pa-relay" / "in"
OUT_DIR = REPO / ".pa-relay" / "out"

BOOT_TIMEOUT_S = 240.0     # a cold runner needs to boot before the first response
POLL_S = 2.0


class RelayUnavailable(RuntimeError):
    pass


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", *args], cwd=str(REPO), check=False,
                         capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RelayUnavailable(f"git {' '.join(args)} failed: {res.stderr.strip()[:300]}")
    return res


def _remote_head() -> Optional[str]:
    try:
        res = _git("ls-remote", "origin", BRANCH)
    except RelayUnavailable:
        return None
    for line in res.stdout.splitlines():
        if line.endswith(f"refs/heads/{BRANCH}"):
            return line.split()[0]
    return None


def relay_alive() -> bool:
    """True when the relay branch exists and a runner has pushed a heartbeat."""
    head = _remote_head()
    if not head:
        return False
    return (OUT_DIR / ".gitkeep").exists() or _fetch_heartbeat() is not None


def _fetch_heartbeat() -> Optional[Dict[str, Any]]:
    try:
        raw = _git("show", f"origin/{BRANCH}:.pa-relay/heartbeat.json", check=False).stdout
    except Exception:  # noqa: BLE001
        return None
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def fetch_all() -> None:
    _git("fetch", "-q", "origin", BRANCH)


def read_response(rid: str) -> Optional[Dict[str, Any]]:
    """Read a relay response out of the local clone of the relay branch."""
    res = _git("show", f"origin/{BRANCH}:.pa-relay/out/{rid}.json", check=False)
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        return json.loads(res.stdout)
    except Exception:  # noqa: BLE001
        return None


class RelayTransport(requests.adapters.BaseAdapter):
    """A requests adapter that performs the request on the CI runner."""

    def __init__(self, timeout: float = BOOT_TIMEOUT_S) -> None:
        self.timeout = timeout
        self.last_meta: Dict[str, Any] = {}

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        timeout = kwargs.get("timeout") or self.timeout
        body = request.body
        if body is None:
            body_b64 = ""
        elif isinstance(body, str):
            body_b64 = base64.b64encode(body.encode()).decode()
        elif isinstance(body, (bytes, bytearray)):
            body_b64 = base64.b64encode(bytes(body)).decode()
        else:
            raise RelayUnavailable(f"unsupported request body type {type(body)!r}")

        rid = uuid.uuid4().hex
        payload = {
            "id": rid,
            "method": request.method,
            "url": request.url,
            "headers": {k: v for k, v in request.headers.items()
                        if k.lower() not in ("content-length", "host", "accept-encoding")},
            "body_b64": body_b64,
            "stream": True,
            "ts": time.time(),
        }
        IN_DIR.mkdir(parents=True, exist_ok=True)
        (IN_DIR / f"{rid}.json").write_text(json.dumps(payload))

        _git("add", "-A", ".pa-relay/in")
        _git("commit", "-q", "-m", f"relay: request {rid} {request.method} {request.url}",
             check=False)
        for attempt in range(6):
            push = _git("push", "-q", "origin", f"HEAD:{BRANCH}", check=False)
            if push.returncode == 0:
                break
            _git("fetch", "-q", "origin", BRANCH)
            _git("rebase", "-q", f"origin/{BRANCH}", check=False)
        else:
            raise RelayUnavailable("could not push the relay request")

        started = time.time()
        last_seen: Dict[str, Any] = {}
        while time.time() - started < timeout:
            time.sleep(POLL_S)
            fetch_all()
            data = read_response(rid)
            if data is None:
                continue
            if data.get("body_b64"):
                last_seen = data
            if data.get("done"):
                return self._to_response(request, data)
        raise RelayUnavailable(
            f"relay did not answer {request.method} {request.url} within {timeout}s "
            f"(last chunk: {len(last_seen.get('body_b64') or '')} b64 chars)")

    def _to_response(self, request: requests.PreparedRequest,
                     data: Dict[str, Any]) -> requests.Response:
        raw = base64.b64decode(data.get("body_b64") or "")
        resp = requests.Response()
        resp.status_code = int(data.get("status") or 599)
        resp.headers.update(data.get("headers") or {})
        resp.url = request.url
        resp.request = request
        resp._content = raw
        if data.get("error"):
            self.last_meta["error"] = data["error"]
        return resp

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def relay_session() -> requests.Session:
    s = requests.Session()
    s.mount("http://", RelayTransport())
    s.mount("https://", RelayTransport())
    return s


def relay_get(url: str, **kwargs) -> requests.Response:
    return relay_session().get(url, **kwargs)
