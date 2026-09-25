"""HTTP surface tests against a real uvicorn server on a real port."""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]


def _client(api) -> httpx.Client:
    return httpx.Client(base_url=api.base, timeout=60.0)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_list_models_shape(api):
    r = _client(api).get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 18
    for card in body["data"]:
        assert set(card) >= {"id", "object", "created", "owned_by", "label", "providers"}
        assert card["object"] == "model"
        assert card["owned_by"] == "pointy-arrow"
        assert isinstance(card["created"], int)
        assert card["created"] <= int(time.time()) + 5
        assert card["providers"], card["id"]
        for prov in card["providers"]:
            assert prov["provider"]
            assert prov["alias"]
            assert prov["id"]
            assert isinstance(prov.get("context_window", 0), int)


def test_models_alias_without_the_v1_prefix(api):
    r = _client(api).get("/models")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 18


def test_retrieve_a_group_model(api):
    r = _client(api).get("/v1/models/kimi-k3")
    assert r.status_code == 200
    card = r.json()
    assert card["id"] == "kimi-k3"
    assert card["label"] == "Kimi K3"
    # "group" is only emitted for pinned (provider-qualified) cards
    assert "group" not in card
    assert len(card["providers"]) >= 1


def test_retrieve_a_pinned_model_with_a_slash_in_the_id(api):
    pinned = "voidai/kimi-k3"
    r = _client(api).get(f"/v1/models/{pinned}")
    assert r.status_code == 200
    card = r.json()
    assert card["id"] == pinned
    assert card["providers"][0]["provider"] == "voidai"


def test_retrieve_by_human_label(api):
    r = _client(api).get("/v1/models/Kimi%20K3")
    assert r.status_code == 200
    assert r.json()["id"] == "kimi-k3"


def test_retrieve_unknown_model_is_a_404(api):
    r = _client(api).get("/v1/models/definitely-not-a-model")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"
    assert "definitely-not-a-model" in err["message"]


def test_retrieve_an_empty_id_is_a_404(api):
    r = _client(api).get("/v1/models/")
    assert r.status_code in (404, 405)


# --------------------------------------------------------------------------- #
# health / reload
# --------------------------------------------------------------------------- #
def test_health_reports_live_credential_stats(api):
    r = _client(api).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["server"] == "pointy-arrow-api"
    assert body["models"] == {"groups": 18, "pinned": 37}
    creds = body["credentials"]
    assert creds["total"] == 2
    assert creds["working"] == 2
    assert creds["available"] == 2
    assert creds["harvesting"] is False
    assert len(creds["credentials"]) == 2
    for c in creds["credentials"]:
        assert c["email"].endswith("@example.test")
        assert c["successes"] == 0
        assert c["failures"] == 0
        assert c["depleted"] is False


def test_admin_reload_reports_counts(api):
    r = _client(api).post("/admin/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["credentials_reloaded"] == 2
    assert body["models_reloaded"] == 55
    assert body["credentials"]["total"] == 2


def test_admin_reload_picks_up_a_credentials_file_rewritten_on_disk(api_factory, tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"credentials": []}))
    extra = api_factory(creds_path=path)
    try:
        assert _client(extra).get("/health").json()["credentials"]["total"] == 0
        path.write_text(json.dumps({"credentials": [
            {"id": "x", "email": "x@y.z", "access_token": "t", "refresh_token": "r",
             "expires_at": time.time() + 9999}
        ]}))
        r = _client(extra).post("/admin/reload")
        assert r.json()["credentials_reloaded"] == 1
        assert _client(extra).get("/health").json()["credentials"]["total"] == 1
    finally:
        extra.stop()


def test_admin_reload_picks_up_a_models_file_rewritten_on_disk(api_factory, tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"models": [
        {"provider": "p", "model": "solo", "pid": "Solo Model", "features": ["chat"]}
    ]}))
    srv = api_factory(models_path=models)
    try:
        r = _client(srv).post("/admin/reload")
        assert r.json()["models_reloaded"] == 2  # 1 group + 1 pinned
        body = _client(srv).get("/v1/models").json()
        assert [c["id"] for c in body["data"]] == ["solo-model"]
        assert _client(srv).get("/health").json()["models"] == {"groups": 1, "pinned": 1}
        # the new model is usable
        chat = _client(srv).post("/v1/chat/completions", json={
            "model": "solo-model", "messages": [{"role": "user", "content": "hi"}]})
        assert chat.status_code in (200, 502)
    finally:
        srv.stop()


def test_a_broken_models_file_degrades_to_no_models(api_factory, tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"version": 1}))  # no "models" key
    srv = api_factory(models_path=models)
    try:
        assert _client(srv).get("/v1/models").json()["data"] == []
        r = _client(srv).post("/v1/chat/completions", json={
            "model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "no_models"
    finally:
        srv.stop()


# --------------------------------------------------------------------------- #
# chat completions: request validation
# --------------------------------------------------------------------------- #
def test_chat_unknown_model_is_a_404(api):
    r = _client(api).post("/v1/chat/completions", json={
        "model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"


def test_chat_requires_a_messages_field(api):
    r = _client(api).post("/v1/chat/completions", json={"model": "kimi-k3"})
    assert r.status_code == 422


def test_chat_rejects_a_non_list_messages_field(api):
    r = _client(api).post("/v1/chat/completions",
                          json={"model": "kimi-k3", "messages": "nope"})
    assert r.status_code == 422


def test_chat_rejects_a_malformed_body(api):
    r = _client(api).post("/v1/chat/completions", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_chat_accepts_a_message_without_content(api):
    """content is optional in the schema; an empty string is sent upstream."""
    r = _client(api).post("/v1/chat/completions",
                          json={"model": "kimi-k3", "messages": [{"role": "user"}]})
    # no credentials scenario configured -> upstream error, but not a 422
    assert r.status_code in (200, 502, 503)


def test_all_three_chat_aliases_exist(api):
    for path in ("/v1/chat/completions", "/chat/completions", "/api/v1/chat/completions"):
        r = _client(api).post(path, json={
            "model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code in (200, 502), (path, r.status_code)


def test_unknown_route_is_a_404(api):
    assert _client(api).get("/nope").status_code == 404
    assert _client(api).post("/v1/embeddings", json={}).status_code == 404
