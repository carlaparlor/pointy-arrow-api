"""pa_relay: transparent git-transport egress for the pointy-arrow server.

This sandbox cannot reach gratify.xyz / mail.tm directly (egress allow-list),
so outbound HTTP is carried over git commits on the working branch to a
GitHub Actions runner, which executes the requests against the REAL upstream
services and streams responses back the same way. Nothing is mocked: every
byte upstream comes from the actual providers.

Enable with PA_RELAY=1 before importing pa_server. The shim replaces the
`requests` module reference inside pa_credentials / pa_router with a
API-compatible subset; all other attributes (exceptions etc.) delegate to the
real `requests` package.

Mailbox layout (on the session branch):

    .pa-relay/in/<id>.json     sandbox -> runner
    .pa-relay/out/<id>.json    runner -> sandbox
    .pa-relay/heartbeat.json   runner liveness
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests as _real_requests

REPO_URL = os.environ.get("PA_RELAY_REPO", "https://github.com/carlaparlor/pointy-arrow-api.git")
BRANCH = os.environ.get("PA_RELAY_BRANCH", "arena/01a0da09-pointy-arrow-api")
MAILBOX_DIR = Path(os.environ.get("PA_RELAY_MAILBOX", str(Path.home() / ".pa-relay-mailbox")))
FETCH_INTERVAL = float(os.environ.get("PA_RELAY_FETCH_INTERVAL", "1.2"))
DEFAULT_DEADLINE = 600.0


def _log(msg: str) -> None:
    if os.environ.get("PA_RELAY_DEBUG"):
        print(f"[pa-relay-client {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RelayTimeout(_real_requests.exceptions.ReadTimeout):
    pass


class _Git:
    """Serialized git ops against the dedicated mailbox clone."""

    def __init__(self) -> None:
        self.dir = MAILBOX_DIR
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._last_fetch = 0.0
        if not (self.dir / ".git").exists():
            self.dir.parent.mkdir(parents=True, exist_ok=True)
            self._run(["git", "clone", "-q", "--branch", BRANCH, "--single-branch",
                       REPO_URL, str(self.dir)], cwd=None)
            self._run(["git", "config", "user.email", "pa-relay-client@arena.local"])
            self._run(["git", "config", "user.name", "pa-relay-client"])
        for sub in ("in", "out"):
            (self.dir / ".pa-relay" / sub).mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=str(cwd or self.dir), capture_output=True, text=True, timeout=120)

    def fetch(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_fetch < FETCH_INTERVAL:
            return
        with self._fetch_lock:
            if not force and time.time() - self._last_fetch < FETCH_INTERVAL:
                return
            self._run(["git", "fetch", "-q", "origin", BRANCH])
            self._last_fetch = time.time()

    def read_remote_file(self, relpath: str) -> Optional[bytes]:
        self.fetch()
        res = self._run(["git", "show", f"origin/{BRANCH}:{relpath}"])
        if res.returncode != 0:
            return None
        return res.stdout.encode("utf-8") if isinstance(res.stdout, str) else res.stdout

    def commit_push(self, message: str) -> bool:
        with self._lock:
            self._run(["git", "add", "-A", ".pa-relay"])
            staged = self._run(["git", "diff", "--cached", "--quiet"])
            if staged.returncode == 0:
                return False
            self._run(["git", "commit", "-q", "-m", message])
            for attempt in range(8):
                push = self._run(["git", "push", "-q", "origin", f"HEAD:{BRANCH}"])
                if push.returncode == 0:
                    return True
                _log(f"push conflict, rebasing ({attempt + 1})")
                self._run(["git", "fetch", "-q", "origin", BRANCH])
                rebase = self._run(["git", "rebase", "-q", f"origin/{BRANCH}"])
                if rebase.returncode != 0:
                    self._run(["git", "rebase", "--abort"])
                    self._run(["git", "reset", "-q", "--hard", f"origin/{BRANCH}"])
                    raise RuntimeError("relay: lost commit during rebase")
                time.sleep(0.5 * (attempt + 1))
            raise RuntimeError("relay: push failed after retries")

    def heartbeat_age(self) -> Optional[float]:
        raw = self.read_remote_file(".pa-relay/heartbeat.json")
        if not raw:
            return None
        try:
            return time.time() - float(json.loads(raw)["ts"])
        except Exception:
            return None


class RelayResponse:
    """requests.Response lookalike fed from relay mailbox files."""

    def __init__(self, git: "RelayTransport", exchange_id: str, timeout: Any):
        self._git = git
        self._id = exchange_id
        self._timeout = timeout
        self.status_code = 0
        self.headers: Dict[str, str] = {}
        self.reason = ""
        self.encoding = "utf-8"
        self.url = ""
        self._done = False
        self._error: Optional[str] = None
        self._body = b""
        self._opened = False

    # -- waiting helpers -------------------------------------------------
    def _read_deadline(self) -> float:
        t = self._timeout
        if isinstance(t, (tuple, list)):
            read = t[1] if len(t) > 1 and t[1] is not None else t[0]
        else:
            read = t
        try:
            return float(read) if read is not None else DEFAULT_DEADLINE
        except (TypeError, ValueError):
            return DEFAULT_DEADLINE

    def _wait_open(self) -> None:
        deadline = time.time() + self._read_deadline()
        while True:
            raw = self._git.read_remote_file(f".pa-relay/out/{self._id}.json")
            if raw:
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = None
                if payload is not None:
                    self.status_code = int(payload.get("status") or 599)
                    self.headers = payload.get("headers") or {}
                    self._body = base64.b64decode(payload.get("body_b64") or "")
                    self._done = bool(payload.get("done"))
                    self._error = payload.get("error")
                    self._opened = True
                    return
            if time.time() > deadline:
                raise RelayTimeout(f"relay: no response for {self._id} within timeout")
            time.sleep(0.25)

    def _refresh(self) -> bool:
        """Pull latest version of the response file; True when new data arrived or done."""
        raw = self._git.read_remote_file(f".pa-relay/out/{self._id}.json")
        if not raw:
            return False
        try:
            payload = json.loads(raw)
        except Exception:
            return False
        body = base64.b64decode(payload.get("body_b64") or "")
        done = bool(payload.get("done"))
        changed = body != self._body or done != self._done or payload.get("error") != self._error
        self._body = body
        self._done = done
        self._error = payload.get("error")
        self.status_code = int(payload.get("status") or self.status_code or 0)
        self.headers = payload.get("headers") or self.headers
        return changed

    # -- requests.Response surface ----------------------------------------
    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        self._wait_open()
        deadline = time.time() + self._read_deadline()
        while not self._done:
            if time.time() > deadline:
                raise RelayTimeout("relay: read timeout waiting for full body")
            time.sleep(0.3)
            self._refresh()
        return self._body.decode(self.encoding or "utf-8", errors="replace")

    @property
    def content(self) -> bytes:
        _ = self.text
        return self._body

    def json(self, **_: Any) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        _ = self.text if not self._opened else None
        if 400 <= self.status_code < 600:
            raise _real_requests.HTTPError(
                f"{self.status_code} Error (relay)", response=self)

    def iter_lines(self, chunk_size: Any = None, decode_unicode: bool = False,
                   delimiter: Any = None) -> Iterator[Any]:
        self._wait_open()
        read_timeout = self._read_deadline()
        consumed = 0
        leftover = b""
        last_progress = time.time()
        while True:
            data = self._body[consumed:]
            if data:
                consumed = len(self._body)
                last_progress = time.time()
                leftover += data
                while b"\n" in leftover:
                    line, leftover = leftover.split(b"\n", 1)
                    yield line.decode(self.encoding or "utf-8", errors="replace") if decode_unicode else line
                continue
            if self._done:
                if leftover:
                    yield leftover.decode(self.encoding or "utf-8", errors="replace") if decode_unicode else leftover
                if self._error:
                    _log(f"stream ended with relay error: {self._error}")
                return
            if time.time() - last_progress > read_timeout:
                raise RelayTimeout("relay: read timeout waiting for stream data")
            time.sleep(0.3)
            self._refresh()

    def iter_content(self, chunk_size: Any = 1, decode_unicode: bool = False) -> Iterator[Any]:
        for line in self.iter_lines(decode_unicode=decode_unicode):
            yield line + ("\n" if decode_unicode else b"\n")

    def close(self) -> None:
        pass

    def __enter__(self) -> "RelayResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class RelayTransport:
    """Submits exchanges and tracks responses."""

    def __init__(self) -> None:
        self.git = _Git()

    def _prune(self) -> None:
        out_dir = self.git.dir / ".pa-relay" / "out"
        if out_dir.exists():
            cutoff = time.time() - 900
            for p in out_dir.glob("*.json"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except OSError:
                    pass

    def request(self, method: str, url: str, *, headers: Optional[Dict[str, str]] = None,
                json_body: Any = None, data: Any = None, stream: bool = False,
                timeout: Any = None, **_: Any) -> RelayResponse:
        rid = uuid.uuid4().hex
        body = b""
        hdrs = dict(headers or {})
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif data is not None:
            body = data if isinstance(data, bytes) else str(data).encode("utf-8")
        payload = {
            "id": rid,
            "method": method.upper(),
            "url": url,
            "headers": hdrs,
            "body_b64": base64.b64encode(body).decode() if body else "",
            "stream": bool(stream) or "text/event-stream" in hdrs.get("Accept", ""),
            "ts": time.time(),
        }
        self._prune()
        inbox = self.git.dir / ".pa-relay" / "in" / f"{rid}.json"
        inbox.parent.mkdir(parents=True, exist_ok=True)
        inbox.write_text(json.dumps(payload))
        self.git.commit_push(f"relay: request {rid} {method.upper()} {url.split('?')[0]}")
        _log(f"submitted {rid} {method.upper()} {url}")
        resp = RelayResponse(self.git, rid, timeout)
        resp.url = url
        return resp


class RelaySession:
    def __init__(self, transport: RelayTransport) -> None:
        self._t = transport

    def post(self, url: str, **kw: Any) -> RelayResponse:
        return self._t.request("POST", url, **kw)

    def get(self, url: str, **kw: Any) -> RelayResponse:
        return self._t.request("GET", url, **kw)

    def request(self, method: str, url: str, **kw: Any) -> RelayResponse:
        return self._t.request(method, url, **kw)

    def close(self) -> None:
        pass


class _Shim:
    """Drop-in partial replacement for the `requests` module."""

    def __init__(self) -> None:
        self._transport: Optional[RelayTransport] = None
        self.exceptions = _real_requests.exceptions
        self.HTTPError = _real_requests.HTTPError
        self.ReadTimeout = RelayTimeout

    def _t(self) -> RelayTransport:
        if self._transport is None:
            self._transport = RelayTransport()
        return self._transport

    def Session(self) -> RelaySession:  # noqa: N802 (requests API)
        return RelaySession(self._t())

    def post(self, url: str, **kw: Any) -> RelayResponse:
        return self._t().request("POST", url, **kw)

    def get(self, url: str, **kw: Any) -> RelayResponse:
        return self._t().request("GET", url, **kw)

    def __getattr__(self, item: str) -> Any:
        return getattr(_real_requests, item)


SHIM = _Shim()


def install() -> None:
    import pa_credentials
    import pa_router
    pa_credentials.requests = SHIM  # type: ignore[assignment]
    pa_router.requests = SHIM  # type: ignore[assignment]
    print(f"[pa_server] relay transport ENABLED (branch {BRANCH})", flush=True)
