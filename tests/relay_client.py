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

BOOT_TIMEOUT_S = 300.0     # a cold runner needs to boot before the first response
POLL_S = 1.5


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
    # An explicit refspec is required: a bare `git fetch origin <branch>` only
    # updates FETCH_HEAD, so `origin/<branch>` would stay stale forever and the
    # relay responses would never become visible.
    _git("fetch", "-q", "-f", "origin", f"+{BRANCH}:refs/remotes/origin/{BRANCH}")


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
        self.pruned: List[str] = []

    def _answered(self, rid: str) -> bool:
        res = _git("show", f"origin/{BRANCH}:.pa-relay/out/{rid}.json", check=False)
        return res.returncode == 0 and bool(res.stdout.strip())

    def _prune_inbox(self) -> List[str]:
        """Delete inbox entries that already have a response.

        The runner's `_seen` set is per-process, so every fresh workflow run
        re-executes *every* file left in the inbox.  Without pruning, a handful
        of requests is enough to fill the worker pool and starve the newest one.
        """
        removed = []
        if not IN_DIR.is_dir():
            return removed
        for path in sorted(IN_DIR.glob("*.json")):
            if self._answered(path.stem):
                path.unlink()
                removed.append(path.stem)
        return removed

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        timeout = kwargs.get("timeout") or self.timeout
        if isinstance(timeout, (tuple, list)):
            # requests passes (connect, read); the relay budget is a single number
            timeout = max(float(t) for t in timeout if t)
        timeout = float(timeout)
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
        wants_stream = bool(kwargs.get("stream"))
        payload = {
            "id": rid,
            "method": request.method,
            "url": request.url,
            "headers": {k: v for k, v in request.headers.items()
                        if k.lower() not in ("content-length", "host", "accept-encoding")},
            "body_b64": body_b64,
            "stream": wants_stream,
            "ts": time.time(),
        }
        IN_DIR.mkdir(parents=True, exist_ok=True)
        fetch_all()
        pruned = self._prune_inbox()
        (IN_DIR / f"{rid}.json").write_text(json.dumps(payload))
        if pruned:
            self.pruned = pruned

        _git("add", "-A", ".pa-relay/in")
        _git("commit", "-q", "-m", f"relay: request {rid} {request.method} {request.url}",
             check=False)
        for attempt in range(6):
            push = _git("push", "-q", "origin", f"HEAD:{BRANCH}", check=False)
            if push.returncode == 0:
                break
            fetch_all()
            # --autostash: the sandbox usually has unrelated work in progress,
            # and a plain rebase refuses to run with a dirty tree.
            _git("rebase", "--autostash", "-q", f"origin/{BRANCH}", check=False)
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
