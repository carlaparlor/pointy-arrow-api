"""Shared fixtures: real processes, real sockets, real files, zero mocks.

The environment overrides (``PA_SUPABASE_URL`` / ``PA_MAILTM_BASE`` /
``PA_CHAT_ENDPOINT`` / ``PA_CREDENTIALS_PATH``) are applied at *conftest import
time*, i.e. before any test module imports ``pa_credentials`` / ``pa_router``,
because those modules capture the endpoints as module-level constants.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import httpx
import pytest
import uvicorn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------- #
# session-wide scratch space + endpoint overrides (must happen at import time)
# --------------------------------------------------------------------------- #
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SESSION_DIR = Path(tempfile.mkdtemp(prefix="pointy-arrow-tests-"))
SESSION_PORT = free_port()
SESSION_BASE = f"http://127.0.0.1:{SESSION_PORT}"
SESSION_SCENARIO = SESSION_DIR / "scenario.json"
SESSION_CREDS = SESSION_DIR / "credentials.json"

# Point every module-level endpoint constant at the local reference upstream.
os.environ["PA_SUPABASE_URL"] = SESSION_BASE
os.environ["PA_MAILTM_BASE"] = SESSION_BASE
os.environ["PA_CHAT_ENDPOINT"] = f"{SESSION_BASE}/api/chat"
os.environ["PA_CREDENTIALS_PATH"] = str(SESSION_CREDS)
os.environ["PA_TEST_SCENARIO"] = str(SESSION_SCENARIO)
os.environ["PA_AUTO_HARVEST"] = "0"


def wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last = "never contacted"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
            last = f"status {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - we want the raw reason
            last = repr(exc)
        time.sleep(0.15)
    raise RuntimeError(f"{url} did not come up: {last}")


# --------------------------------------------------------------------------- #
# scenario file (drives the reference upstream)
# --------------------------------------------------------------------------- #
class Scenario:
    """Reads/writes the JSON file that scripts the reference upstream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.write({})

    def write(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def _merge(self, key: str, value: Any) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data[key] = value
        self.write(data)

    def set_chat(self, entries: List[Dict[str, Any]]) -> None:
        self._merge("chat", entries)

    def set_auth(self, entries: List[Dict[str, Any]]) -> None:
        self._merge("auth", entries)

    def set_signup(self, entries: List[Dict[str, Any]]) -> None:
        self._merge("signup", entries)

    def set_mail(self, **kw: Any) -> None:
        self._merge("mail", kw)

    def reset(self) -> None:
        self.write({})


# --------------------------------------------------------------------------- #
# reference upstream (real uvicorn server)
# --------------------------------------------------------------------------- #
class ReferenceUpstream:
    def __init__(self, port: int, scenario: Scenario) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.scenario = scenario
        self._saved_env: Dict[str, Optional[str]] = {}
        self.server = uvicorn.Server(
            uvicorn.Config("reference_upstream:app", host="127.0.0.1", port=port,
                           log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    # -- env management so a per-test upstream does not leak into later tests --
    def _push_env(self) -> None:
        for key, value in (
            ("PA_TEST_SCENARIO", str(self.scenario.path)),
            ("PA_SUPABASE_URL", self.base),
            ("PA_MAILTM_BASE", self.base),
            ("PA_CHAT_ENDPOINT", f"{self.base}/api/chat"),
        ):
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _pop_env(self) -> None:
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_env.clear()

    def start(self) -> "ReferenceUpstream":
        self._push_env()
        self.thread.start()
        wait_http(f"{self.base}/__log")
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self._pop_env()

    # -- introspection over real HTTP --
    def log(self) -> List[Dict[str, Any]]:
        return httpx.get(f"{self.base}/__log", timeout=10).json()["entries"]

    def reset(self) -> None:
        httpx.post(f"{self.base}/__reset", timeout=10)
        self.scenario.reset()

    def chat_requests(self) -> List[Dict[str, Any]]:
        return [e for e in self.log() if e["path"] == "/api/chat"]

    def auth_requests(self) -> List[Dict[str, Any]]:
        return [e for e in self.log() if e["path"].startswith("/auth/v1/")]

    def mail_requests(self) -> List[Dict[str, Any]]:
        return [
            e for e in self.log()
            if e["path"] in ("/domains", "/accounts", "/token", "/messages")
            or e["path"].startswith("/messages/")
        ]


# --------------------------------------------------------------------------- #
# the server under test (real uvicorn subprocess, exactly like production)
# --------------------------------------------------------------------------- #
class ApiServer:
    upstream_base: str = ""

    def __init__(self, port: int, env: Dict[str, str], log_path: Path) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        full_env = dict(os.environ)
        full_env.update(env)
        full_env["PA_AUTO_HARVEST"] = "0"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pa_server:app",
             "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"],
            cwd=str(REPO), env=full_env,
            stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        )

    def start(self) -> "ApiServer":
        wait_http(f"{self.base}/health")
        return self

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def logs(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def write_credentials(
    path: Path, count: int = 2, *, expired: bool = False, source: str = "test",
    tokens: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    creds = []
    for i in range(count):
        creds.append(
            {
                "id": str(uuid.uuid4()),
                "email": f"user{i}@example.test",
                "password": f"pw-{i}",
                "access_token": (tokens[i] if tokens and i < len(tokens) else f"access-token-{i}"),
                "refresh_token": f"refresh-token-{i}",
                "expires_at": (time.time() - 600) if expired else (time.time() + 86400),
                "user_agent": "test-agent/1.0",
                "source": source,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "count": len(creds), "credentials": creds}, indent=2),
        encoding="utf-8",
    )
    return creds


def make_api(
    tmp_path: Path,
    upstream: ReferenceUpstream,
    *,
    creds_path: Optional[Path] = None,
    models_path: Optional[Path] = None,
    creds: int = 2,
    expired: bool = False,
    tokens: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> ApiServer:
    creds_path = creds_path or (tmp_path / "credentials.json")
    if not creds_path.exists():
        write_credentials(creds_path, count=creds, expired=expired, tokens=tokens)
    env = {
        "PA_CHAT_ENDPOINT": f"{upstream.base}/api/chat",
        "PA_SUPABASE_URL": upstream.base,
        "PA_MAILTM_BASE": upstream.base,
        "PA_CREDENTIALS_PATH": str(creds_path),
        "PA_AUTO_HARVEST": "0",
        "PA_AUTO_REFRESH": "0",
    }
    if models_path is not None:
        env["PA_MODELS_PATH"] = str(models_path)
    if extra_env:
        env.update(extra_env)
    server = ApiServer(free_port(), env, tmp_path / f"api-{uuid.uuid4().hex[:6]}.log")
    server.upstream_base = upstream.base
    server.start()
    return server


def read_credentials(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("credentials", [])


def run_python(code: str, env: Optional[Dict[str, str]] = None, timeout: float = 180.0):
    """Run a snippet in a *real* subprocess with a bespoke environment."""
    full = dict(os.environ)
    full.setdefault("PYTHONPATH", str(REPO))
    if env:
        full.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO), env=full, capture_output=True, text=True, timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def session_upstream() -> Iterator[ReferenceUpstream]:
    """One reference upstream for the whole session, used by the library tests."""
    up = ReferenceUpstream(SESSION_PORT, Scenario(SESSION_SCENARIO)).start()
    yield up
    up.stop()


@pytest.fixture(autouse=True)
def _clean_upstream(session_upstream: ReferenceUpstream) -> Iterator[None]:
    """Every test starts from a blank scenario and an empty request log."""
    session_upstream.reset()
    yield


@pytest.fixture()
def scenario(upstream: ReferenceUpstream) -> Scenario:
    """The scenario file of the reference upstream the server under test uses."""
    return upstream.scenario


@pytest.fixture()
def upstream(tmp_path: Path, session_upstream: ReferenceUpstream) -> Iterator[ReferenceUpstream]:
    """An isolated reference upstream (own port + own scenario) for chat tests."""
    isolated = ReferenceUpstream(free_port(), Scenario(tmp_path / "scenario.json"))
    isolated.start()
    try:
        yield isolated
    finally:
        isolated.stop()


@pytest.fixture()
def api(tmp_path: Path, upstream: ReferenceUpstream) -> Iterator[ApiServer]:
    creds_path = tmp_path / "credentials.json"
    write_credentials(creds_path, count=2)
    server = make_api(tmp_path, upstream, creds_path=creds_path)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture()
def api_factory(tmp_path: Path, upstream: ReferenceUpstream):
    """Start additional servers with a bespoke environment (all still real)."""
    started: List[ApiServer] = []

    def _make(**kwargs: Any) -> ApiServer:
        srv = make_api(tmp_path, upstream, **kwargs)
        started.append(srv)
        return srv

    yield _make
    for srv in started:
        srv.stop()


@pytest.fixture()
def creds_path() -> Path:
    return SESSION_CREDS
