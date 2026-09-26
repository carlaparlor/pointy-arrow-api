"""Regression tests for the crash/atomicity bugs found while testing."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pa_credentials import Credential, CredentialFile
from pa_server import _normalize_usage


# --------------------------------------------------------------------------- #
# usage normalisation must never raise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("usage", [
    {"prompt_tokens": "abc", "completion_tokens": "xyz"},
    {"prompt_tokens": None, "completion_tokens": []},
    {"prompt_tokens": {"a": 1}},
    {"prompt_tokens": "12", "completion_tokens": "7"},
    {"prompt_tokens": 3.9, "completion_tokens": 2.1},
    {"prompt_tokens": True},
    "not a dict",
])
def test_normalize_usage_survives_junk(usage):
    out = _normalize_usage(usage, "abcd", "efgh")
    assert set(out) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert out["total_tokens"] == out["prompt_tokens"] + out["completion_tokens"]
    assert all(isinstance(v, int) for v in out.values())


def test_normalize_usage_prefers_the_first_present_key():
    assert _normalize_usage({"prompt_tokens": 1, "input_tokens": 99}, "x", "y")["prompt_tokens"] == 1
    assert _normalize_usage({"input_tokens": 5}, "x", "y")["prompt_tokens"] == 5
    assert _normalize_usage({"candidatesTokenCount": 6}, "x", "y")["completion_tokens"] == 6


def test_normalize_usage_falls_back_to_estimates():
    out = _normalize_usage({}, "a" * 40, "b" * 80)
    assert out == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_normalize_usage_handles_none():
    out = _normalize_usage(None, "abcd", "efgh")
    assert out["prompt_tokens"] == 1 and out["completion_tokens"] == 1


# --------------------------------------------------------------------------- #
# concurrent saves must not corrupt the pool file
# --------------------------------------------------------------------------- #
def _pool(path: Path, tag: str) -> None:
    for i in range(40):
        CredentialFile(path).save([
            Credential(id=f"{tag}-{i}", email=f"{tag}{i}@x.test",
                       access_token=f"t-{tag}-{i}", refresh_token=f"r-{tag}-{i}",
                       expires_at=float(i)),
        ])
        time.sleep(0.001)


def test_concurrent_saves_never_corrupt_the_file(tmp_path: Path):
    path = tmp_path / "pool.json"
    threads = [threading.Thread(target=_pool, args=(path, f"w{i}")) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    # the file must always be complete, parseable JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["credentials"][0]["access_token"].startswith("t-")
    # and no temp files may be left behind
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_save_creates_missing_parent_directories(tmp_path: Path):
    path = tmp_path / "deep" / "nested" / "pool.json"
    CredentialFile(path).save([Credential(id="a", access_token="t", refresh_token="r")])
    assert CredentialFile(path).load()[0].id == "a"


def test_a_reader_never_sees_a_partial_file(tmp_path: Path):
    """Read while writers hammer the same path."""
    path = tmp_path / "pool.json"
    stop = threading.Event()
    errors: List[str] = []

    def writer(tag: str) -> None:
        i = 0
        while not stop.is_set():
            CredentialFile(path).save([
                Credential(id=f"{tag}-{i}", access_token="x" * 200, refresh_token="y" * 200,
                           expires_at=float(i)),
            ])
            i += 1

    def reader() -> None:
        while not stop.is_set():
            if not path.exists():
                time.sleep(0.001)  # not written yet; that is not a partial read
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                assert data["count"] == 1
                assert len(data["credentials"][0]["access_token"]) == 200
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
                return

    writers = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(3)]
    for t in writers + readers:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in writers + readers:
        t.join(timeout=30)
    assert errors == [], errors


# --------------------------------------------------------------------------- #
# a pool file full of junk still loads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [
    json.dumps({"credentials": ["a", 1, None, [], {}]}),
    json.dumps({"credentials": "nope"}),
    json.dumps([{"id": "ok"}, "junk"]),
    json.dumps({"credentials": [{"id": "x", "expires_at": "not-a-number"}]}),
    json.dumps({"credentials": [{"id": "x", "expires_at": None}]}),
])
def test_pool_file_with_junk_rows_still_loads(tmp_path: Path, payload):
    path = tmp_path / "pool.json"
    path.write_text(payload, encoding="utf-8")
    creds = CredentialFile(path).load()
    assert all(isinstance(c, Credential) for c in creds)
    assert all(c.expires_at == 0.0 for c in creds if c.id in ("x",))
