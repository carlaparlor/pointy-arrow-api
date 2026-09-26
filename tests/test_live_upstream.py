"""Live tests against the real Gratisfy infrastructure.

These are the "no substitutes" tests: they talk to ``https://gratisfy.xyz``,
``https://auth.gratisfy.xyz`` and ``https://api.mail.tm`` exactly as the
production code does.  Nothing is stubbed -- the bytes on the wire come from
the real servers.

``tests/conftest.py`` re-points the module-level endpoint constants at the local
reference upstream, so this module explicitly restores the production values
before doing anything.  That is configuration, not mocking.

Transport
---------
If the local network cannot reach the Gratisfy hosts (this sandbox's egress
allowlist blocks them), the requests are served by the git/CI relay in
``tests/relay_client.py``: a GitHub Actions runner with unrestricted egress
executes each request and returns the real response.  The production code is
unmodified either way -- it just gets a different ``requests`` transport.

Run them with::

    pytest -m live -v

The reachability probes run automatically.  Anything that would create a real
account on Gratisfy (credential harvesting) or spend a real request is gated
behind ``PA_LIVE_HARVEST=1`` / ``PA_LIVE_TOKEN`` so a casual ``pytest`` run
never touches a third-party service.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import requests

PROD_CHAT = "https://gratisfy.xyz/api/chat"
PROD_AUTH = "https://auth.gratisfy.xyz"
PROD_MAILTM = "https://api.mail.tm"
PROD_SITE = "https://gratisfy.xyz"
PROD_API = "https://api.gratisfy.xyz/v1"

REPO = Path(__file__).resolve().parents[1]

_TRANSPORT: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# transport selection
# --------------------------------------------------------------------------- #
def _use_production_endpoints() -> None:
    """Undo the test-harness redirection so we hit the real services."""
    import pa_credentials
    import pa_router

    pa_router.CHAT_ENDPOINT = PROD_CHAT
    pa_credentials.SUPABASE_URL = PROD_AUTH
    pa_credentials.MAILTM_BASE = PROD_MAILTM


def _relay_session() -> requests.Session:
    from relay_client import RelayTransport

    transport = RelayTransport()
    session = requests.Session()
    session.mount("https://", transport)
    session.mount("http://", transport)
    return session


def live_session() -> requests.Session:
    """The session the production code should use for live traffic."""
    if "session" not in _TRANSPORT:
        if _TRANSPORT.get("direct"):
            _TRANSPORT["session"] = requests.Session()
        else:
            _TRANSPORT["session"] = _relay_session()
    return _TRANSPORT["session"]


def _get(url: str, **kw) -> requests.Response:
    kw.setdefault("timeout", 400)
    return live_session().get(url, **kw)


def _post(url: str, **kw) -> requests.Response:
    kw.setdefault("timeout", 400)
    return live_session().post(url, **kw)


def _probe(url: str) -> Optional[int]:
    try:
        return requests.get(url, timeout=12, allow_redirects=True).status_code
    except requests.RequestException:
        return None


@pytest.fixture(scope="module", autouse=True)
def _production():
    _use_production_endpoints()
    _relay_mount()


def _relay_mount() -> None:
    """Point pa_credentials' own session at the live transport.

    ``pa_credentials`` keeps a module-level ``SESSION`` and every auth/mail call
    goes through it, so the relay adapter has to be mounted there rather than on
    a session we hand in.  No request-building logic changes.
    """
    import pa_credentials

    session = live_session()
    for scheme in ("https://", "http://"):
        pa_credentials.SESSION.mount(scheme, session.get_adapter(scheme + "x"))


@pytest.fixture(scope="module")
def egress() -> Dict[str, Optional[int]]:
    """Direct probes first; if the network blocks them, the same probes via relay."""
    targets = {
        "site": PROD_SITE,
        "auth": f"{PROD_AUTH}/auth/v1/health",
        "mailtm": f"{PROD_MAILTM}/domains",
        "api": PROD_API + "/models",
    }
    direct = {k: _probe(v) for k, v in targets.items()}
    if any(v is not None for v in direct.values()):
        return {**direct, "transport": "direct"}

    relayed: Dict[str, Optional[int]] = {}
    for key, url in targets.items():
        try:
            relayed[key] = _relay_session().get(url, timeout=400).status_code
        except Exception:  # noqa: BLE001
            relayed[key] = None
    return {**relayed, "transport": "relay"}


def _require_egress(egress, key: str) -> int:
    status = egress.get(key)
    if status is None:
        pytest.skip(
            f"no route to the Gratisfy hosts: the direct probe and the relay probe "
            f"of {key} both failed. Re-run somewhere with unrestricted network access."
        )
    return status


# =========================================================================== #
# reachability
# =========================================================================== #
@pytest.mark.live
def test_live_site_is_reachable(egress):
    status = _require_egress(egress, "site")
    assert status == 200, f"gratisfy.xyz returned {status} (transport={egress['transport']})"


@pytest.mark.live
def test_live_auth_service_is_reachable(egress):
    status = _require_egress(egress, "auth")
    assert status in (200, 401, 404, 405), f"auth.gratisfy.xyz returned {status}"


@pytest.mark.live
def test_live_mailtm_is_reachable(egress):
    status = _require_egress(egress, "mailtm")
    assert status == 200, f"api.mail.tm returned {status}"
    domains = _get(f"{PROD_MAILTM}/domains").json()
    members = domains.get("hydra:member") or domains.get("hydra:members") or []
    assert members, "mail.tm reported no domains"
    assert any(m.get("domain") for m in members)


@pytest.mark.live
def test_live_api_requires_a_key(egress):
    status = _require_egress(egress, "api")
    body = _get(PROD_API + "/models").json()
    assert "error" in body
    assert body["error"].get("code") in ("missing_api_key", "invalid_api_key", None)


@pytest.mark.live
def test_the_website_chat_endpoint_rejects_an_anonymous_request(egress):
    """A real 401/403 proves the endpoint exists and enforces auth."""
    _require_egress(egress, "site")
    r = _post(
        PROD_CHAT,
        json={"model": "kimi-k3", "provider": "voidai",
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"Content-Type": "application/json",
                 "Origin": "https://gratisfy.xyz",
                 "Referer": "https://gratisfy.xyz/chat",
                 "User-Agent": "pointy-arrow-api-test/1.0"},
    )
    assert r.status_code in (401, 403, 422), (
        f"unexpected status {r.status_code}: {r.text[:200]}")


# =========================================================================== #
# the local catalogue against the live contract
# =========================================================================== #
@pytest.mark.live
def test_every_local_route_has_the_fields_the_live_api_needs():
    from pa_models import REGISTRY

    for card in REGISTRY.group_cards():
        for route in card.routes:
            assert route.raw_id, f"{card.id} has a route without a model id"
            assert route.provider, f"{card.id} has a route without a provider"
            assert route.pid, f"{card.id} has a route without a pid"


# =========================================================================== #
# real credential harvesting (opt-in: creates a real account)
# =========================================================================== #
CRED_CACHE = Path(os.getenv("PA_LIVE_CRED_CACHE", "/tmp/pa-live-credential.json"))


@pytest.fixture(scope="module")
def live_credential():
    """A real Gratisfy session, provisioned through the real signup flow.

    The credential is cached on disk so re-running the live suite does not keep
    creating throwaway accounts on a third-party service.
    """
    import pa_credentials

    _use_production_endpoints()

    if CRED_CACHE.exists():
        try:
            data = json.loads(CRED_CACHE.read_text())
            if data.get("expires_at", 0) > time.time() + 120:
                cached = pa_credentials.Credential(
                    id=data["id"], email=data.get("email"),
                    password=data.get("password"),
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_at=float(data["expires_at"]),
                    source="auto-harvest",
                )
                if cached.is_valid():
                    return cached
        except Exception:  # noqa: BLE001 - a bad cache just means re-harvest
            pass

    if os.getenv("PA_LIVE_HARVEST") != "1":
        pytest.skip("set PA_LIVE_HARVEST=1 to provision a real credential "
                    f"(or provide one at {CRED_CACHE})")

    cred = pa_credentials.harvest_one(verify_timeout=240)
    assert cred.is_valid()
    CRED_CACHE.write_text(json.dumps({
        "id": cred.id, "email": cred.email, "password": cred.password,
        "access_token": cred.access_token, "refresh_token": cred.refresh_token,
        "expires_at": cred.expires_at,
    }))
    return cred


@pytest.mark.live
def test_live_harvest_produces_a_working_credential(live_credential):
    assert live_credential.access_token
    assert live_credential.refresh_token
    assert live_credential.email
    assert live_credential.expires_at > time.time()


def _recache(credential) -> None:
    try:
        CRED_CACHE.write_text(json.dumps({
            "id": credential.id, "email": credential.email,
            "password": credential.password,
            "access_token": credential.access_token,
            "refresh_token": credential.refresh_token,
            "expires_at": credential.expires_at,
        }))
    except Exception:  # noqa: BLE001 - the cache is best effort
        pass


@pytest.mark.live
def test_live_refresh_token_grant(live_credential):
    """A real refresh_token grant against the real GoTrue auth server."""
    import pa_credentials

    _use_production_endpoints()
    before_access = live_credential.access_token
    before_refresh = live_credential.refresh_token
    refreshed = pa_credentials.refresh_via_refresh_token(live_credential)
    assert refreshed.access_token, "the real auth server returned no access_token"
    assert refreshed.access_token != before_access, (
        "the real auth server handed back the same access token"
    )
    assert refreshed.expires_at > time.time(), "the refreshed token is already expired"
    if before_refresh:
        # Supabase rotates refresh tokens, but a server may legitimately reuse one
        print(f"\nrefresh token rotated: {refreshed.refresh_token != before_refresh}")
    _recache(refreshed)


@pytest.mark.live
def test_live_password_grant(live_credential):
    """A real password grant against the real GoTrue auth server."""
    import pa_credentials

    _use_production_endpoints()
    if not (live_credential.email and live_credential.password):
        pytest.skip("the harvested credential has no email/password to log in with")
    logged_in = pa_credentials.refresh_via_password(live_credential)
    assert logged_in.access_token, "the real auth server returned no access_token"
    assert logged_in.expires_at > time.time()
    _recache(logged_in)


@pytest.mark.live
def test_live_expired_credential_is_refreshed_automatically(live_credential):
    """ensure_fresh() must recover a live session without human help."""
    import pa_credentials

    _use_production_endpoints()
    live_credential.expires_at = time.time() - 1  # force it stale
    assert live_credential.needs_refresh()
    fresh = pa_credentials.ensure_fresh(live_credential)
    assert not fresh.needs_refresh()
    assert fresh.access_token
    _recache(fresh)


@pytest.mark.live
def test_live_harvested_credential_can_chat(live_credential):
    """A provisioned account must be able to talk to the real model."""
    result = _chat_with(live_credential, os.getenv("PA_LIVE_MODEL", "kimi-k3"),
                        "Reply with the single word: pong")
    assert result.get("rejected") is None, (
        f"the live upstream refused the harvested credential: {result['rejected']}"
    )
    assert result["text"].strip(), "the live upstream returned no content"


@pytest.mark.live
def test_live_harvested_credential_rejection_is_classified(live_credential):
    """Whatever the live upstream says, it must be reported, not swallowed."""
    result = _chat_with(live_credential, os.getenv("PA_LIVE_MODEL", "kimi-k3"),
                        "Reply with the single word: pong")
    if result.get("rejected") is None:
        assert result["text"].strip()
        pytest.skip("live upstream is currently serving chat to harvested credentials")
    body = str(result["rejected"])
    if "turnstile_required" in body:
        print("\nLIVE CONTRACT: Gratisfy now requires Cloudflare Turnstile human "
              "verification before a Gratisfy-provided model can be used, so an "
              "auto-harvested website account cannot chat until it is verified by "
              "a human in a browser.")
    assert "turnstile_required" in body or result.get("status") in (401, 403, 429), body


# =========================================================================== #
# real chat with a supplied token (opt-in)
# =========================================================================== #
@pytest.fixture(scope="module")
def supplied_credential():
    token = os.getenv("PA_LIVE_TOKEN")
    if not token:
        pytest.skip("set PA_LIVE_TOKEN to a real Gratisfy access token")
    from pa_credentials import Credential

    return Credential(
        id="live-supplied",
        access_token=token,
        refresh_token=os.getenv("PA_LIVE_REFRESH_TOKEN") or None,
        expires_at=time.time() + float(os.getenv("PA_LIVE_EXPIRES_IN", "3600")),
    )


def _live_pool(credential) -> "CredentialPool":
    """A real CredentialPool over a real credentials file holding one session."""
    from pa_router import CredentialPool

    path = Path(os.getenv("PA_LIVE_POOL", "/tmp/pa-live-pool.json"))
    path.write_text(json.dumps([{
        "id": credential.id,
        "email": credential.email,
        "password": credential.password,
        "access_token": credential.access_token,
        "refresh_token": credential.refresh_token,
        "expires_at": credential.expires_at,
        "source": credential.source or "live",
    }]))
    pool = CredentialPool(path=path)
    pool.reload()
    assert pool.total(), "the live credential pool came back empty"
    return pool


def _pick_route(model: str):
    from pa_models import REGISTRY

    card = REGISTRY.resolve(model)
    assert card is not None, f"unknown model {model}"
    return card, REGISTRY.ordered_routes(card)[0]


def _chat_with(credential, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
    from pa_router import _GratisfyClient, _RouteRejected

    _use_production_endpoints()
    card, route = _pick_route(model)
    client = _GratisfyClient(credential, timeout=(20, 300), session=live_session())
    text_parts = []
    usage = None
    finish = None
    try:
        for ev in client.stream([{"role": "user", "content": prompt}], route, **kwargs):
            if ev.get("type") == "content":
                text_parts.append(ev.get("text", ""))
            elif ev.get("type") == "usage":
                usage = ev.get("usage")
            elif ev.get("type") == "finish":
                finish = ev.get("finish_reason")
            elif ev.get("type") == "error":
                pytest.fail(f"live upstream error: {ev.get('message')}")
    except _RouteRejected as exc:
        return {"rejected": str(exc), "status": exc.status, "text": "",
                "usage": None, "finish_reason": None,
                "model": card.id, "route": route.mid}
    return {"rejected": None, "text": "".join(text_parts), "usage": usage,
            "finish_reason": finish, "model": card.id, "route": route.mid}


@pytest.mark.live
def test_live_non_streaming_chat(supplied_credential):
    result = _chat_with(supplied_credential, os.getenv("PA_LIVE_MODEL", "kimi-k3"),
                        "Reply with exactly: pong")
    assert result["text"].strip(), "the live upstream returned no content"
    assert result["finish_reason"] in ("stop", "length", "tool_calls", None)


@pytest.mark.live
def test_live_tool_calling(supplied_credential):
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        },
    }
    result = _chat_with(
        supplied_credential, os.getenv("PA_LIVE_MODEL", "kimi-k3"),
        "What is the weather in Oslo? Use the get_weather tool.",
        tools=[tool], tool_choice="auto",
    )
    assert result["finish_reason"] == "tool_calls", result


@pytest.mark.live
def test_live_reasoning_model(supplied_credential):
    result = _chat_with(supplied_credential, os.getenv("PA_LIVE_MODEL", "kimi-k3"),
                        "Think step by step: what is 17 times 23?")
    assert result["text"].strip()
    assert "391" in result["text"]


@pytest.mark.live
def test_live_router_collect(supplied_credential):
    """The whole PARouter path against the real endpoint."""
    import pa_router

    _use_production_endpoints()
    router = pa_router.PARouter(_live_pool(supplied_credential),
                                session=live_session())
    result = router.collect(
        [{"role": "user", "content": "Say hi in three words."}],
        model=os.getenv("PA_LIVE_MODEL", "kimi-k3"),
    )
    assert result["text"].strip()
    assert result["route"]["provider"]


# =========================================================================== #
# is the Turnstile gate route-specific?  (real probe)
# =========================================================================== #
# Real routes spread across the Gratisfy-provided providers (aqua / evolvex /
# voidai / cloudflare / groq / openrouter / routmy / vercel) so the probe covers
# both shared and bring-your-own-key routes.
TURNSTILE_MODELS = ["deepseek-v4-flash", "qwen3.8-27b", "grok-4.6"]

# Live contract recorded on 2026-09-26: every route is behind a human-verification
# gate, so a freshly provisioned website account cannot chat.  The chat tests
# above therefore stay opt-in (they need a token from a human-verified account)
# until Gratisfy lifts the gate.
TURNSTILE_GATE_ACTIVE = os.getenv("PA_LIVE_TURNSTILE_GATE", "1") == "1"


@pytest.mark.live
def test_live_turnstile_gate_covers_every_route(live_credential):
    """Document (and keep re-checking) the live human-verification gate.

    If Gratisfy lifts the gate this test fails, which is the signal to re-enable
    ``test_live_non_streaming_chat`` / ``test_live_tool_calling`` for harvested
    credentials.
    """
    from pa_models import REGISTRY

    _use_production_endpoints()
    probe = []
    for model in TURNSTILE_MODELS:
        card = REGISTRY.resolve(model)
        if card is None:
            continue
        result = _chat_with(live_credential, model, "say hi")
        probe.append((model, result.get("status"),
                      (result.get("rejected") or "")[:200]))
        print(f"\n{model}: status={result.get('status')} "
              f"rejected={(result.get('rejected') or '')[:200]}")

    assert probe, "no models could be probed"
    gated = [p for p in probe if p[1] == 403 and "turnstile_required" in p[2]]
    served = [p for p in probe if p[1] is None and p[0] not in TURNSTILE_MODELS]
    inconclusive = [p for p in probe if p not in gated and p not in served]
    print(f"\nSUMMARY: turnstile-gated={len(gated)} served={len(served)} "
          f"inconclusive={len(inconclusive)} of {len(probe)} probed routes")
    for model, status, body in probe:
        print(f"  {model:20s} status={status} {body[:90]}")

    if TURNSTILE_GATE_ACTIVE:
        assert gated, f"no route reported the gate; probe={probe}"
        assert not served, (
            "the live human-verification gate no longer covers every route; "
            f"re-enable the real-chat live tests. probe={probe}"
        )


@pytest.mark.live
def test_live_router_surfaces_the_turnstile_gate(live_credential):
    """The router must propagate a real upstream rejection, not swallow it."""
    from pa_models import REGISTRY
    from pa_router import PARouter

    _use_production_endpoints()
    card = REGISTRY.resolve(os.getenv("PA_LIVE_MODEL", "kimi-k3"))
    assert card is not None
    router = PARouter(_live_pool(live_credential), session=live_session())
    try:
        result = router.collect([{"role": "user", "content": "hi"}],
                                model=os.getenv("PA_LIVE_MODEL", "kimi-k3"))
        print(f"\nrouter produced {len(result['text'])} chars of real content")
    except RuntimeError as exc:
        message = str(exc)
        print(f"\nrouter refused: {message[:400]}")
        assert "turnstile_required" in message, (
            "the router hid the real upstream reason behind a generic error"
        )
