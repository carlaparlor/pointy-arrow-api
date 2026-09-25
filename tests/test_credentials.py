"""Credential-pool tests.

Everything here talks to a real HTTP server (the reference upstream in
``tests/reference_upstream.py``) over a real TCP socket, or manipulates real
files on disk.  No mocks, no monkeypatching, no fakes injected into the code
under test.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pa_credentials import (
    Credential,
    CredentialFile,
    add_credentials,
    ensure_fresh,
    harvest,
    harvest_one,
    import_from_supabase_localstorage,
    refresh,
)

from conftest import read_credentials, run_python


# --------------------------------------------------------------------------- #
# the Credential value object
# --------------------------------------------------------------------------- #
def test_credential_validity_requires_both_tokens():
    assert Credential(id="1", access_token="a", refresh_token="r").is_valid()
    assert not Credential(id="1", access_token="a").is_valid()
    assert not Credential(id="1", refresh_token="r").is_valid()
    assert not Credential(id="1").is_valid()


def test_credential_without_access_token_always_needs_refresh():
    assert Credential(id="1").needs_refresh()


def test_needs_refresh_honours_the_60s_leeway():
    soon = Credential(id="1", access_token="a", expires_at=time.time() + 30)
    assert soon.needs_refresh()
    later = Credential(id="1", access_token="a", expires_at=time.time() + 600)
    assert not later.needs_refresh()


def test_bearer_header_and_missing_token_error():
    cred = Credential(id="1", access_token="tok")
    assert cred.bearer_header() == "Bearer tok"
    with pytest.raises(RuntimeError):
        Credential(id="1").bearer_header()


def test_update_token_keeps_existing_refresh_token_when_absent():
    cred = Credential(id="1", access_token="old", refresh_token="keep")
    cred.update_token(access_token="new", expires_in=3600)
    assert cred.access_token == "new"
    assert cred.refresh_token == "keep"
    assert cred.expires_at > time.time() + 3000
    assert cred.last_refreshed_at


def test_update_token_floors_short_expiries_at_30s():
    cred = Credential(id="1", access_token="old")
    cred.update_token(access_token="new", expires_in=5)
    assert cred.expires_at <= time.time() + 30


def test_update_token_without_expires_in_leaves_expiry_alone():
    cred = Credential(id="1", access_token="old", expires_at=123.0)
    cred.update_token(access_token="new")
    assert cred.expires_at == 123.0


# --------------------------------------------------------------------------- #
# CredentialFile -- real files on disk
# --------------------------------------------------------------------------- #
def test_credential_file_round_trip(tmp_path: Path):
    path = tmp_path / "creds.json"
    creds = [
        Credential(id="a", email="a@x.test", password="p1", access_token="t1",
                   refresh_token="r1", expires_at=999.0, source="user"),
        Credential(id="b", email="b@x.test", access_token="t2", refresh_token="r2",
                   expires_at=1000.0, source="auto-harvest"),
    ]
    CredentialFile(path).save(creds)
    on_disk = json.loads(path.read_text())
    assert on_disk["count"] == 2
    assert on_disk["version"] == 1
    assert on_disk["updated_at"]

    loaded = CredentialFile(path).load()
    assert [c.id for c in loaded] == ["a", "b"]
    assert loaded[0].email == "a@x.test"
    assert loaded[0].password == "p1"
    assert loaded[0].access_token == "t1"
    assert loaded[0].expires_at == 999.0
    assert loaded[1].source == "auto-harvest"
    assert loaded[1].is_valid()


def test_credential_file_load_missing_file(tmp_path: Path):
    assert CredentialFile(tmp_path / "nope.json").load() == []


def test_credential_file_load_corrupt_file(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{{{not json", encoding="utf-8")
    assert CredentialFile(path).load() == []


def test_credential_file_accepts_a_bare_list(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([{"id": "x", "access_token": "t", "refresh_token": "r"}]))
    loaded = CredentialFile(path).load()
    assert len(loaded) == 1 and loaded[0].id == "x"
    assert loaded[0].source == "user"


def test_credential_file_ignores_unreadable_rows(tmp_path: Path):
    path = tmp_path / "junk.json"
    path.write_text(json.dumps({"credentials": ["nope", 5, None, {"id": "ok"}]}))
    loaded = CredentialFile(path).load()
    assert [c.id for c in loaded] == ["ok"]


def test_credential_file_defaults_expires_at_to_zero(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"credentials": [{"id": "x"}]}))
    assert CredentialFile(path).load()[0].expires_at == 0.0


# --------------------------------------------------------------------------- #
# refresh against a real auth server
# --------------------------------------------------------------------------- #
@pytest.fixture()
def auth_env(tmp_path: Path, session_upstream):
    path = tmp_path / "creds.json"
    return {
        "PA_SUPABASE_URL": session_upstream.base,
        "PA_MAILTM_BASE": session_upstream.base,
        "PA_CREDENTIALS_PATH": str(path),
        "path": path,
    }


def test_ensure_fresh_is_a_noop_for_a_valid_credential(auth_env, session_upstream):
    cred = Credential(id="1", access_token="a", refresh_token="r",
                      expires_at=time.time() + 3600)
    assert ensure_fresh(cred) is cred
    assert session_upstream.auth_requests() == []


def test_refresh_via_refresh_token_hits_the_real_endpoint(auth_env, session_upstream):
    session_upstream.scenario.set_auth([{"access_token": "fresh-access", "expires_in": 7200}])
    cred = Credential(id="1", email="u@x.test", password="pw", access_token="old",
                      refresh_token="rt", expires_at=time.time() - 10)
    refresh(cred)
    assert cred.access_token == "fresh-access"
    assert cred.expires_at > time.time() + 7000

    reqs = session_upstream.auth_requests()
    assert len(reqs) == 1
    assert reqs[0]["path"] == "/auth/v1/token"
    assert reqs[0]["query"] == "grant_type=refresh_token"
    assert reqs[0]["body"]["refresh_token"] == "rt"


def test_refresh_falls_back_to_password_grant_on_400(auth_env, session_upstream):
    session_upstream.scenario.set_auth([
        {"status": 400, "body": json.dumps({"error": "invalid refresh token"})},
        {"access_token": "pw-access", "refresh_token": "pw-refresh", "expires_in": 1800},
    ])
    cred = Credential(id="1", email="u@x.test", password="secret", access_token="old",
                      refresh_token="stale", expires_at=time.time() - 10)
    refresh(cred)
    assert cred.access_token == "pw-access"
    assert cred.refresh_token == "pw-refresh"

    reqs = session_upstream.auth_requests()
    assert [r["query"] for r in reqs] == ["grant_type=refresh_token", "grant_type=password"]
    assert reqs[1]["body"] == {"email": "u@x.test", "password": "secret"}


def test_refresh_does_not_fall_back_on_server_error(auth_env, session_upstream):
    session_upstream.scenario.set_auth([{"status": 500, "body": "boom"}])
    cred = Credential(id="1", email="u@x.test", password="secret", access_token="old",
                      refresh_token="rt", expires_at=time.time() - 10)
    with pytest.raises(Exception):
        refresh(cred)
    assert [r["query"] for r in session_upstream.auth_requests()] == ["grant_type=refresh_token"]


def test_refresh_uses_password_grant_when_there_is_no_refresh_token(auth_env, session_upstream):
    session_upstream.scenario.set_auth([{"access_token": "pw-only", "expires_in": 600}])
    cred = Credential(id="1", email="u@x.test", password="secret", access_token="old",
                      expires_at=time.time() - 10)
    refresh(cred)
    assert cred.access_token == "pw-only"
    assert cred.refresh_token == "refresh-pw"
    assert [r["query"] for r in session_upstream.auth_requests()] == ["grant_type=password"]


def test_refresh_raises_when_nothing_can_refresh():
    with pytest.raises(RuntimeError):
        refresh(Credential(id="1", access_token="old", expires_at=0.0))


def test_ensure_fresh_propagates_a_failing_refresh(auth_env, session_upstream):
    session_upstream.scenario.set_auth([{"status": 503, "body": "down"}])
    cred = Credential(id="1", access_token="old", refresh_token="rt", expires_at=0.0)
    with pytest.raises(Exception):
        ensure_fresh(cred)


# --------------------------------------------------------------------------- #
# harvest_one -- full signup flow over real HTTP
# --------------------------------------------------------------------------- #
def test_harvest_one_end_to_end(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_auth([
        {"access_token": "harvest-access", "refresh_token": "harvest-refresh", "expires_in": 3600},
    ])
    cred = harvest_one(verify_timeout=10)
    assert cred.is_valid()
    assert cred.source == "auto-harvest"
    assert cred.access_token == "harvest-access"
    assert cred.refresh_token == "harvest-refresh"
    assert cred.email.endswith("@harvest.test")
    assert cred.password
    assert cred.expires_at > time.time() + 3000
    assert cred.id

    paths = [(r["method"], r["path"]) for r in session_upstream.log()]
    assert ("GET", "/domains") in paths
    assert ("POST", "/accounts") in paths
    assert ("POST", "/token") in paths
    assert ("POST", "/auth/v1/signup") in paths
    assert ("POST", "/auth/v1/token") in paths
    assert ("GET", "/auth/v1/verify") in paths  # the confirmation link was followed
    assert ("GET", "/messages") in paths
    assert ("GET", "/messages/msg-1") in paths

    signup = next(r for r in session_upstream.log() if r["path"] == "/auth/v1/signup")
    assert signup["body"]["email"] == cred.email
    assert signup["body"]["password"] == cred.password
    verify = next(r for r in session_upstream.log() if r["path"] == "/auth/v1/verify")
    assert verify["query"] == "token=verify-token-abc&type=signup"


def test_harvest_one_extracts_the_verify_url_from_html_only(tmp_path: Path, session_upstream):
    """The link lives in an HTML attribute with &amp; entities."""
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_auth([{"access_token": "a", "expires_in": 60}])
    cred = harvest_one(verify_timeout=10)
    assert cred.access_token == "a"
    verify = next(r for r in session_upstream.log() if r["path"] == "/auth/v1/verify")
    assert "type=signup" in verify["query"]


def test_harvest_one_honours_the_domain_override(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_auth([{"access_token": "a", "expires_in": 60}])
    cred = harvest_one("forced.test", verify_timeout=10)
    assert cred.email.endswith("@forced.test")
    # the domain endpoint must not have been consulted at all
    assert ("GET", "/domains") not in [(r["method"], r["path"]) for r in session_upstream.log()]


def test_harvest_one_raises_when_signup_fails(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_signup([{"status": 422, "body": "email already registered"}])
    with pytest.raises(Exception):
        harvest_one(verify_timeout=5)


def test_harvest_one_raises_when_no_verify_email_arrives(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test", verify_path="/auth/v1/nope")
    session_upstream.scenario.set_auth([{"access_token": "a", "expires_in": 60}])
    with pytest.raises(RuntimeError, match="no verify email"):
        harvest_one(verify_timeout=2)


# --------------------------------------------------------------------------- #
# harvest() -- batch, writes real files
# --------------------------------------------------------------------------- #
def test_harvest_writes_credentials_to_disk(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_auth([
        {"access_token": f"a{i}", "refresh_token": f"r{i}", "expires_in": 3600} for i in range(3)
    ])
    out = tmp_path / "pool.json"
    events = []
    got = harvest(3, output=out, on_event=lambda k, i, n, d: events.append((k, i, d)),
                  verify_timeout=10)
    assert len(got) == 3
    assert [c.access_token for c in got] == ["a0", "a1", "a2"]
    assert all(e[0] == "ok" for e in events)
    stored = read_credentials(out)
    assert [c["access_token"] for c in stored] == ["a0", "a1", "a2"]
    assert all(c["source"] == "auto-harvest" for c in stored)


def test_harvest_appends_to_an_existing_pool(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_auth([{"access_token": "new", "expires_in": 60}])
    out = tmp_path / "pool.json"
    out.write_text(json.dumps({"credentials": [
        {"id": "old", "access_token": "old-access", "refresh_token": "old-refresh",
         "email": "old@x.test", "expires_at": time.time() + 9999, "source": "user"}
    ]}))
    harvest(1, output=out, verify_timeout=10)
    stored = read_credentials(out)
    assert [c["id"] for c in stored][0] == "old"
    assert stored[-1]["access_token"] == "new"


def test_harvest_reports_and_survives_failures(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="harvest.test")
    session_upstream.scenario.set_signup([{"status": 500, "body": "nope"}])
    out = tmp_path / "pool.json"
    events = []
    got = harvest(2, output=out, on_event=lambda k, i, n, d: events.append((k, i, d)),
                  verify_timeout=5)
    assert got == []
    assert [e[0] for e in events] == ["error", "error"]
    assert not out.exists()


# --------------------------------------------------------------------------- #
# importing sessions by hand
# --------------------------------------------------------------------------- #
def test_import_from_supabase_localstorage_accepts_dict_and_json():
    payload = {"access_token": "a", "refresh_token": "r", "expires_at": 123.5,
               "user": {"email": "me@x.test"}}
    c1 = import_from_supabase_localstorage(payload)
    c2 = import_from_supabase_localstorage(json.dumps(payload))
    for c in (c1, c2):
        assert c.access_token == "a" and c.refresh_token == "r"
        assert c.email == "me@x.test"
        assert c.expires_at == 123.5
        assert c.source == "browser-paste"


@pytest.mark.parametrize("payload", [
    {"refresh_token": "r"},
    {"access_token": "a"},
    {},
    "not json at all",
])
def test_import_from_supabase_localstorage_rejects_bad_payloads(payload):
    with pytest.raises(Exception):
        import_from_supabase_localstorage(payload)


def test_add_credentials_appends_browser_paste_and_email_password(tmp_path: Path):
    out = tmp_path / "pool.json"
    out.write_text(json.dumps({"credentials": [
        {"id": "old", "access_token": "a", "refresh_token": "r", "expires_at": 1.0}
    ]}))
    add_credentials(
        [
            {"access_token": "a2", "refresh_token": "r2", "expires_at": 2.0},
            {"email": "e@x.test", "password": "pw"},
        ],
        output=out,
    )
    stored = read_credentials(out)
    assert len(stored) == 3
    assert stored[1]["access_token"] == "a2"
    assert stored[1]["source"] == "browser-paste"
    assert stored[2]["email"] == "e@x.test"
    assert stored[2]["source"] == "email-pass"


def test_add_credentials_with_an_int_is_a_noop(tmp_path: Path):
    out = tmp_path / "pool.json"
    out.write_text(json.dumps({"credentials": [{"id": "old", "access_token": "a",
                                                "refresh_token": "r"}]}))
    assert [c.id for c in add_credentials(3, output=out)] == ["old"]


# --------------------------------------------------------------------------- #
# the CLI -- a real process
# --------------------------------------------------------------------------- #
def test_cli_list_on_an_empty_pool(tmp_path: Path):
    env = {"PA_CREDENTIALS_PATH": str(tmp_path / "none.json")}
    proc = run_python("import sys; sys.argv=['pa_credentials','list']; import pa_credentials; raise SystemExit(pa_credentials.main())", env)
    assert proc.returncode == 0, proc.stderr
    assert "loaded 0 credential(s)" in proc.stdout


def test_cli_list_reports_expiry_state(tmp_path: Path):
    path = tmp_path / "pool.json"
    write = (
        "import json,time;from pathlib import Path;"
        "Path(%r).write_text(json.dumps({'credentials':[{'id':'x','email':'a@b.c',"
        "'access_token':'t','refresh_token':'r','expires_at':time.time()-10}]}))" % str(path)
    )
    run_python(write)
    proc = run_python("import sys; sys.argv=['pa_credentials','list']; import pa_credentials; raise SystemExit(pa_credentials.main())",
                      {"PA_CREDENTIALS_PATH": str(path)})
    assert proc.returncode == 0, proc.stderr
    assert "loaded 1 credential(s)" in proc.stdout
    assert "needs_refresh=True" in proc.stdout


def test_cli_refresh_reaches_the_real_auth_server(tmp_path: Path, session_upstream):
    path = tmp_path / "pool.json"
    run_python(
        "import json,time;from pathlib import Path;"
        "Path(%r).write_text(json.dumps({'credentials':[{'id':'x','email':'a@b.c',"
        "'password':'pw','access_token':'t','refresh_token':'r','expires_at':time.time()-10}]}))" % str(path)
    )
    session_upstream.scenario.set_auth([{"access_token": "cli-access", "refresh_token": "cli-refresh",
                                 "expires_in": 7200}])
    proc = run_python(
        "import sys; sys.argv=['pa_credentials','refresh']; import pa_credentials; raise SystemExit(pa_credentials.main())",
        {"PA_CREDENTIALS_PATH": str(path), "PA_SUPABASE_URL": session_upstream.base},
    )
    assert proc.returncode == 0, proc.stderr
    assert "refreshed 1/1 credential(s)" in proc.stdout
    stored = read_credentials(path)[0]
    assert stored["access_token"] == "cli-access"
    assert stored["refresh_token"] == "cli-refresh"
    assert stored["expires_at"] > time.time() + 5000


def test_cli_refresh_reports_failures_without_crashing(tmp_path: Path, session_upstream):
    path = tmp_path / "pool.json"
    run_python(
        "import json,time;from pathlib import Path;"
        "Path(%r).write_text(json.dumps({'credentials':[{'id':'x','email':'a@b.c',"
        "'access_token':'t','refresh_token':'r','expires_at':time.time()-10}]}))" % str(path)
    )
    session_upstream.scenario.set_auth([{"status": 500, "body": "boom"}])
    proc = run_python(
        "import sys; sys.argv=['pa_credentials','refresh']; import pa_credentials; raise SystemExit(pa_credentials.main())",
        {"PA_CREDENTIALS_PATH": str(path), "PA_SUPABASE_URL": session_upstream.base},
    )
    assert proc.returncode == 0, proc.stderr
    assert "refresh failed" in proc.stdout
    assert "refreshed 0/1" in proc.stdout


def test_cli_harvest_end_to_end(tmp_path: Path, session_upstream):
    session_upstream.scenario.set_mail(domain="cli.test")
    session_upstream.scenario.set_auth([{"access_token": "cli-harvest", "refresh_token": "r",
                                 "expires_in": 3600}])
    out = tmp_path / "pool.json"
    proc = run_python(
        "import sys; sys.argv=['pa_credentials','harvest','-n','1']; import pa_credentials; raise SystemExit(pa_credentials.main())",
        {"PA_CREDENTIALS_PATH": str(out), "PA_SUPABASE_URL": session_upstream.base,
         "PA_MAILTM_BASE": session_upstream.base},
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "1/1 ok" in proc.stdout
    assert "harvested 1 credential(s)" in proc.stdout
    assert read_credentials(out)[0]["access_token"] == "cli-harvest"


def test_cli_with_no_subcommand_defaults_to_list(tmp_path: Path):
    proc = run_python("import sys; sys.argv=['pa_credentials']; import pa_credentials; raise SystemExit(pa_credentials.main())",
                      {"PA_CREDENTIALS_PATH": str(tmp_path / "none.json")})
    assert proc.returncode == 0, proc.stderr
    assert "loaded 0 credential(s)" in proc.stdout
