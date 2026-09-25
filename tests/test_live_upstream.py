"""Live tests against the real Gratisfy infrastructure.

These are the "no substitutes" tests: they talk to ``https://gratisfy.xyz``,
``https://auth.gratisfy.xyz`` and ``https://api.mail.tm`` exactly as the
production code does.

``tests/conftest.py`` re-points the module-level endpoint constants at the local
reference upstream, so this module explicitly restores the production values
before doing anything.  That is configuration, not mocking -- no behaviour is
substituted anywhere.

Run them with::

    pytest -m live -v

The reachability probes run automatically.  Anything that would create a real
account on Gratisfy (credential harvesting) or spend a real request is gated
behind ``PA_LIVE_HARVEST=1`` / ``PA_LIVE_TOKEN`` so a casual ``pytest`` run never
touches a third-party service.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests

PROD_CHAT = "https://gratisfy.xyz/api/chat"
PROD_AUTH = "https://auth.gratisfy.xyz"
PROD_MAILTM = "https://api.mail.tm"
PROD_SITE = "https://gratisfy.xyz"
PROD_API = "https://api.gratisfy.xyz/v1"

REPO = Path(__file__).resolve().parents[1]


def _use_production_endpoints() -> None:
    """Undo the test-harness redirection so we hit the real services."""
    import pa_credentials
    import pa_router

    pa_router.CHAT_ENDPOINT = PROD_CHAT
    pa_credentials.SUPABASE_URL = PROD_AUTH
    pa_credentials.MAILTM_BASE = PROD_MAILTM


@pytest.fixture(scope="module", autouse=True)
def _production():
    _use_production_endpoints()


def _probe(url: str, timeout: float = 12.0) -> Optional[int]:
    """Return the HTTP status, or None when the host is unreachable."""
    try:
        return requests.get(url, timeout=timeout, allow_redirects=True).status_code
    except requests.RequestException:
        return None


@pytest.fixture(scope="module")
def egress() -> Dict[str, Optional[int]]:
    return {
        "site": _probe(PROD_SITE),
        "auth": _probe(f"{PROD_AUTH}/auth/v1/health"),
        "mailtm": _probe(f"{PROD_MAILTM}/domains"),
        "api": _probe(PROD_API + "/models"),
    }


def _require_egress(egress, key: str) -> int:
    status = egress.get(key)
    if status is None:
        pytest.skip(
            f"this sandbox blocks egress to the Gratisfy hosts "
            f"(probe of {key} failed). Re-run somewhere with unrestricted network "
            f"access: pytest -m live -v"
        )
    return status


# =========================================================================== #
# reachability
# =========================================================================== #
@pytest.mark.live
def test_live_site_is_reachable(egress):
    status = _require_egress(egress, "site")
    assert status == 200, f"gratisfy.xyz returned {status}"


@pytest.mark.live
def test_live_auth_service_is_reachable(egress):
    status = _require_egress(egress, "auth")
    assert status in (200, 401, 404, 405), f"auth.gratisfy.xyz returned {status}"


@pytest.mark.live
def test_live_mailtm_is_reachable(egress):
    status = _require_egress(egress, "mailtm")
    assert status == 200, f"api.mail.tm returned {status}"
    domains = requests.get(f"{PROD_MAILTM}/domains", timeout=20).json()
    members = domains.get("hydra:member") or domains.get("hydra:members") or []
    assert members, "mail.tm reported no domains"
    assert any(m.get("domain") for m in members)


@pytest.mark.live
def test_live_api_requires_a_key(egress):
    status = _require_egress(egress, "api")
    body = requests.get(PROD_API + "/models", timeout=20).json()
    assert "error" in body
    assert body["error"].get("code") in ("missing_api_key", "invalid_api_key", None)


# =========================================================================== #
# the local catalogue is well formed against the live contract
# =========================================================================== #
@pytest.mark.live
def test_every_local_route_has_the_fields_the_live_api_needs():
    from pa_models import REGISTRY

    for card in REGISTRY.group_cards():
        for route in card.routes:
            assert route.raw_id, f"{card.id} has a route without a model id"
            assert route.provider, f"{card.id} has a route without a provider"
            assert route.pid, f"{card.id} has a route without a pid"


@pytest.mark.live
def test_the_website_chat_endpoint_rejects_an_anonymous_request(egress):
    """A real 401/403 proves the endpoint exists and enforces auth."""
    _require_egress(egress, "site")
    r = requests.post(
        PROD_CHAT,
        json={"model": "kimi-k3", "provider": "voidai",
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"Content-Type": "application/json",
                 "Origin": "https://gratisfy.xyz",
                 "Referer": "https://gratisfy.xyz/chat",
                 "User-Agent": "pointy-arrow-api-test/1.0"},
        timeout=30,
    )
    assert r.status_code in (401, 403, 422), (
        f"unexpected status {r.status_code}: {r.text[:200]}")


# =========================================================================== #
# real credential harvesting (opt-in: creates a real account)
# =========================================================================== #
@pytest.fixture(scope="module")
def live_credential():
    if os.getenv("PA_LIVE_HARVEST") != "1":
        pytest.skip("set PA_LIVE_HARVEST=1 to provision a real credential")
    import pa_credentials

    _use_production_endpoints()
    cred = pa_credentials.harvest_one()
    assert cred.is_valid()
    return cred


@pytest.mark.live
def test_live_harvest_produces_a_working_credential(live_credential):
    assert live_credential.access_token
    assert live_credential.refresh_token
    assert live_credential.email
    assert live_credential.expires_at > time.time()


@pytest.mark.live
def test_live_harvested_credential_can_chat(live_credential):
    _chat_with(live_credential, "kimi-k3", "Reply with the single word: pong")


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


def _pick_route(model: str):
    from pa_models import REGISTRY

    card = REGISTRY.resolve(model)
    assert card is not None, f"unknown model {model}"
    return card, REGISTRY.ordered_routes(card)[0]


def _chat_with(credential, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
    from pa_router import _GratisfyClient

    _use_production_endpoints()
    card, route = _pick_route(model)
    client = _GratisfyClient(credential, timeout=(15, 120))
    text_parts: List[str] = []
    usage = None
    finish = None
    for ev in client.stream([{"role": "user", "content": prompt}], route, **kwargs):
        if ev.get("type") == "content":
            text_parts.append(ev.get("text", ""))
        elif ev.get("type") == "usage":
            usage = ev.get("usage")
        elif ev.get("type") == "finish":
            finish = ev.get("finish_reason")
        elif ev.get("type") == "error":
            pytest.fail(f"live upstream error: {ev.get('message')}")
    return {"text": "".join(text_parts), "usage": usage,
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
    router = pa_router.PARouter()
    result = router.collect(
        [{"role": "user", "content": "Say hi in three words."}],
        model=os.getenv("PA_LIVE_MODEL", "kimi-k3"),
    )
    assert result["text"].strip()
    assert result["route"]["provider"]
