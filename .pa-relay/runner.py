"""pa-relay runner: executes real HTTP requests queued via git and streams
responses back through git commits on the same branch.

Runs inside a GitHub Actions job (full internet egress). Mailbox layout:

    .pa-relay/in/<id>.json     sandbox -> runner:  {method,url,headers,body_b64,stream}
    .pa-relay/out/<id>.json    runner -> sandbox:  {status,headers,body_b64,done,error}
    .pa-relay/heartbeat.json   runner liveness

The runner is a resident loop: it stays alive while there is activity and
exits when idle, so each new request commit wakes a fresh workflow run.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

REPO_DIR = Path(os.environ.get("PA_RELAY_DIR", os.getcwd())).resolve()
BRANCH = os.environ["PA_RELAY_BRANCH"]
IN_DIR = REPO_DIR / ".pa-relay" / "in"
OUT_DIR = REPO_DIR / ".pa-relay" / "out"
HEARTBEAT = REPO_DIR / ".pa-relay" / "heartbeat.json"

MAX_LIFE_S = 320 * 60          # leave margin under the 360min job timeout
IDLE_EXIT_S = 120              # exit when nothing happened for this long
POLL_S = 1.0                   # remote-check cadence
STREAM_PUSH_S = 1.5            # min seconds between stream update pushes
HEARTBEAT_BUSY_S = 45          # heartbeat cadence while workers are active
HEARTBEAT_IDLE_S = 150         # heartbeat cadence when idle
MAX_WORKERS = 4
HTTP_TIMEOUT = (15, 300)

_git_lock = threading.Lock()
_seen: set[str] = set()
_active = 0
_active_lock = threading.Lock()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_DIR, check=check,
        capture_output=True, text=True,
    )


def log(msg: str) -> None:
    print(f"[pa-relay {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sync_remote() -> None:
    """Fetch the branch and hard-reset to it (all local state is committed)."""
    git("fetch", "-q", "origin", BRANCH)
    git("reset", "-q", "--hard", f"origin/{BRANCH}")
    git("clean", "-q", "-fd", ".pa-relay/in", ".pa-relay/out")


def commit_and_push(message: str) -> bool:
    """Commit everything under .pa-relay and push, rebasing through conflicts."""
    with _git_lock:
        git("add", "-A", ".pa-relay")
        staged = git("diff", "--cached", "--quiet", check=False)
        if staged.returncode == 0:
            return False  # nothing to commit
        git("commit", "-q", "-m", message)
        for attempt in range(8):
            push = git("push", "-q", "origin", f"HEAD:{BRANCH}", check=False)
            if push.returncode == 0:
                return True
            git("fetch", "-q", "origin", BRANCH)
            rebase = git("rebase", "-q", f"origin/{BRANCH}", check=False)
            if rebase.returncode != 0:
                git("rebase", "--abort", check=False)
                git("reset", "-q", "--hard", f"origin/{BRANCH}")
                return False  # lost our change; the other side probably won
            time.sleep(0.4 * (attempt + 1))
        return False


def write_response(rid: str, payload: dict, push: bool = True) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f".{rid}.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(OUT_DIR / f"{rid}.json")
    if push:
        commit_and_push(f"relay: response {rid} {'(done)' if payload.get('done') else '(chunk)'}")


def run_exchange(req: dict) -> None:
    global _active
    rid = req.get("id") or "?"
    try:
        method = (req.get("method") or "GET").upper()
        url = req["url"]
        headers = {k: v for k, v in (req.get("headers") or {}).items()
                   if k.lower() not in ("content-length", "host", "accept-encoding")}
        body = base64.b64decode(req["body_b64"]) if req.get("body_b64") else None
        stream = bool(req.get("stream")) or "text/event-stream" in headers.get("Accept", "")
        log(f"-> {method} {url} stream={stream}")
        try:
            resp = requests.request(method, url, headers=headers, data=body,
                                    stream=stream, timeout=HTTP_TIMEOUT,
                                    allow_redirects=True)
        except Exception as exc:
            log(f"!! {rid} transport error: {exc}")
            write_response(rid, {"status": 599, "headers": {}, "body_b64": "",
                                 "done": True, "error": f"{type(exc).__name__}: {exc}"})
            return
        out_headers = {k: v for k, v in resp.headers.items()
                       if k.lower() not in ("transfer-encoding", "content-encoding",
                                            "content-length", "connection")}
        if not stream:
            data = resp.content
            log(f"<- {rid} {resp.status_code} {len(data)}B")
            write_response(rid, {"status": resp.status_code, "headers": out_headers,
                                 "body_b64": base64.b64encode(data).decode(),
                                 "done": True, "error": None})
            return
        buf = bytearray()
        last_push = time.time()
        write_response(rid, {"status": resp.status_code, "headers": out_headers,
                             "body_b64": "", "done": False, "error": None}, push=False)
        commit_and_push(f"relay: open stream {rid} ({resp.status_code})")
        try:
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    buf.extend(chunk)
                    now = time.time()
                    if now - last_push >= STREAM_PUSH_S:
                        write_response(rid, {"status": resp.status_code, "headers": out_headers,
                                             "body_b64": base64.b64encode(bytes(buf)).decode(),
                                             "done": False, "error": None})
                        last_push = now
        except Exception as exc:
            log(f"!! {rid} stream error: {exc}")
            write_response(rid, {"status": resp.status_code, "headers": out_headers,
                                 "body_b64": base64.b64encode(bytes(buf)).decode(),
                                 "done": True, "error": f"{type(exc).__name__}: {exc}"})
            return
        log(f"<- {rid} stream complete {len(buf)}B")
        write_response(rid, {"status": resp.status_code, "headers": out_headers,
                             "body_b64": base64.b64encode(bytes(buf)).decode(),
                             "done": True, "error": None})
    finally:
        with _active_lock:
            _active -= 1


def main() -> int:
    start = time.time()
    last_activity = time.time()
    last_heartbeat = 0.0
    git("config", "user.email", "pa-relay[bot]@users.noreply.github.com")
    git("config", "user.name", "pa-relay[bot]")
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"runner up on branch {BRANCH}")
    pool = []
    while time.time() - start < MAX_LIFE_S:
        now = time.time()
        if now - last_activity > IDLE_EXIT_S:
            log("idle timeout; exiting")
            break
        try:
            sync_remote()
        except Exception as exc:
            log(f"sync error: {exc}")
            time.sleep(2)
            continue
        for path in sorted(IN_DIR.glob("*.json")):
            rid = path.stem
            if rid in _seen:
                continue
            try:
                req = json.loads(path.read_text())
            except Exception:
                continue
            _seen.add(rid)
            last_activity = time.time()
            with _active_lock:
                busy = _active >= MAX_WORKERS
            if busy:
                log(f"worker pool full; deferring {rid}")
                _seen.discard(rid)
                continue
            with _active_lock:
                _active += 1
            path.unlink(missing_ok=True)
            t = threading.Thread(target=run_exchange, args=(req,), daemon=True)
            t.start()
            pool.append(t)
        pool = [t for t in pool if t.is_alive()]
        with _active_lock:
            busy = _active > 0
        hb_interval = HEARTBEAT_BUSY_S if busy else HEARTBEAT_IDLE_S
        if busy:
            last_activity = time.time()
        if now - last_heartbeat >= hb_interval:
            last_heartbeat = now
            HEARTBEAT.write_text(json.dumps({"ts": now, "busy": busy}))
            commit_and_push("relay: heartbeat")
        time.sleep(POLL_S)
    log("runner exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
