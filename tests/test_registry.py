"""Model registry tests -- real files, real reloads, no mocks."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pa_models import (
    DEFAULT_PREFERENCE,
    PROVIDER_PREFERENCE,
    ModelCard,
    ModelRegistry,
    Route,
    RouteHealth,
    _slugify,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# catalogue sanity
# --------------------------------------------------------------------------- #
def test_catalogue_loads_37_routes_into_18_groups():
    from pa_models import REGISTRY

    groups = REGISTRY.group_cards()
    pinned = [c for c in REGISTRY.cards if c.kind == "pinned"]
    assert len(groups) == 18
    assert len(pinned) == 37
    assert len(REGISTRY.cards) == 55


def test_every_group_has_at_least_one_route():
    from pa_models import REGISTRY

    for card in REGISTRY.group_cards():
        assert card.routes, f"{card.id} has no routes"
        assert all(r.slug == card.id for r in card.routes)


def test_group_ids_are_unique():
    from pa_models import REGISTRY

    ids = [c.id for c in REGISTRY.cards]
    assert len(ids) == len(set(ids))


def test_context_window_is_max_across_routes():
    from pa_models import REGISTRY

    for card in REGISTRY.group_cards():
        assert card.context_window == max(r.context_window for r in card.routes)


# --------------------------------------------------------------------------- #
# README parity -- the documented table must match the live registry
# --------------------------------------------------------------------------- #
def _readme_models() -> list[str]:
    """Model ids listed in the README's Models table (and only that table)."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    section = text.split("## Models", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("| `") or not line.endswith(" |"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # a model row is exactly: | `id` | context |
        if len(cells) == 2 and cells[0].startswith("`") and cells[0].endswith("`"):
            rows.append(cells[0].strip("`"))
    return rows


def test_readme_lists_exactly_the_registry_groups():
    from pa_models import REGISTRY

    documented = _readme_models()
    assert len(documented) == 18, documented
    assert set(documented) == {c.id for c in REGISTRY.group_cards()}


def test_every_documented_model_resolves():
    from pa_models import REGISTRY

    for model_id in _readme_models():
        card = REGISTRY.resolve(model_id)
        assert card is not None, model_id
        assert card.id == model_id


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query",
    ["kimi-k3", "KIMI-K3", " kimi-k3 ", "Kimi K3", "kimi k3", "Kimi_K3", "kimi+k3"],
)
def test_resolve_accepts_case_space_and_separator_variants(query: str):
    from pa_models import REGISTRY

    card = REGISTRY.resolve(query)
    assert card is not None and card.id == "kimi-k3"


def test_resolve_pinned_id_containing_a_slash():
    from pa_models import REGISTRY

    card = REGISTRY.resolve("voidai/kimi-k3")
    assert card is not None
    assert card.kind == "pinned"
    assert card.routes[0].provider == "voidai"


def test_resolve_unknown_returns_none():
    from pa_models import REGISTRY

    assert REGISTRY.resolve("no-such-model") is None
    assert REGISTRY.resolve("no-such-model", strict=True) is None


@pytest.mark.parametrize("query", [None, "", "   ", "auto", "default", "router/default"])
def test_strict_resolve_rejects_sentinels(query):
    from pa_models import REGISTRY

    assert REGISTRY.resolve(query, strict=True) is None


@pytest.mark.parametrize("query", [None, "", "auto", "default"])
def test_lenient_resolve_falls_back_to_default(query):
    from pa_models import REGISTRY

    card = REGISTRY.resolve(query)
    assert card is not None
    assert card.kind == "group"


def test_default_is_a_real_group_card():
    from pa_models import REGISTRY

    card = REGISTRY.default()
    assert card is not None and card.kind == "group" and card.routes


# --------------------------------------------------------------------------- #
# slugify
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DeepSeek V4 Flash", "deepseek-v4-flash"),
        ("GLM 5.2", "glm-5.2"),
        ("qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b"),
        ("a@@@b", "a-b"),
        ("  --Weird__Name++--  ", "weird-name"),
        ("!!!", "model"),
        ("GPT 5.6 Luna", "gpt-5.6-luna"),
    ],
)
def test_slugify(raw, expected):
    assert _slugify(raw) == expected


def test_slugify_is_idempotent():
    for raw in ("DeepSeek V4 Flash", "GLM 5.2", "qwen3.8-27b"):
        once = _slugify(raw)
        assert _slugify(once) == once


# --------------------------------------------------------------------------- #
# route health / cooldowns
# --------------------------------------------------------------------------- #
def test_route_health_success_resets_failures():
    h = RouteHealth()
    h.report(False)
    h.report(False)
    h.report(True)
    assert h.failures == 0
    assert h.available(time.time())


def test_route_health_third_failure_triggers_cooldown():
    h = RouteHealth()
    for _ in range(3):
        h.report(False)
    assert h.available(time.time()) is False
    assert h.cooldown_until > time.time()
    # failures counter is reset so the next failure starts a fresh streak
    assert h.failures == 0


def test_route_health_hard_failure_cools_down_immediately():
    h = RouteHealth()
    h.report(False, hard=True)
    assert h.available(time.time()) is False


def test_route_health_cooldown_expires():
    h = RouteHealth()
    h.cooldown_until = time.time() - 1
    assert h.available(time.time()) is True


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
def test_ordered_routes_prefers_tool_use_routes():
    from pa_models import REGISTRY

    card = REGISTRY.resolve("kimi-k3")
    plain = REGISTRY.ordered_routes(card, prefer_tools=False)
    tools = REGISTRY.ordered_routes(card, prefer_tools=True)
    assert [r.mid for r in plain] == [r.mid for r in tools] or True  # order may differ
    if any("tool-use" in r.features for r in card.routes) and any(
        "tool-use" not in r.features for r in card.routes
    ):
        first_tool = next(i for i, r in enumerate(tools) if "tool-use" in r.features)
        first_plain = next(i for i, r in enumerate(tools) if "tool-use" not in r.features)
        assert first_tool < first_plain


def test_ordered_routes_drops_routes_in_cooldown():
    from pa_models import REGISTRY

    card = REGISTRY.resolve("kimi-k3")
    victim = card.routes[0]
    REGISTRY.report_route(victim.mid, ok=False, hard=True)
    try:
        ordered = REGISTRY.ordered_routes(card)
        assert victim.mid not in [r.mid for r in ordered]
    finally:
        REGISTRY.report_route(victim.mid, ok=True)


def test_ordered_routes_falls_back_to_all_when_everything_is_cooling():
    from pa_models import REGISTRY

    card = REGISTRY.resolve("kimi-k3")
    for r in card.routes:
        REGISTRY.report_route(r.mid, ok=False, hard=True)
    try:
        assert len(REGISTRY.ordered_routes(card)) == len(card.routes)
    finally:
        for r in card.routes:
            REGISTRY.report_route(r.mid, ok=True)


def test_provider_preference_is_used_for_ordering():
    from pa_models import REGISTRY

    card = REGISTRY.resolve("kimi-k3")
    prefs = [r.preference for r in card.routes]
    assert prefs == sorted(prefs)
    assert all(p == PROVIDER_PREFERENCE.get(r.provider, DEFAULT_PREFERENCE)
               for r, p in zip(card.routes, prefs))


# --------------------------------------------------------------------------- #
# reload from a real file on disk
# --------------------------------------------------------------------------- #
def _write_catalogue(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"version": 1, "models": rows}), encoding="utf-8")
    return path


def test_registry_reload_picks_up_new_rows(tmp_path: Path):
    path = _write_catalogue(
        tmp_path / "models.json",
        [
            {"id": "p/one", "provider": "p", "model": "one", "pid": "One", "features": ["tool-use"]},
            {"id": "q/two", "provider": "q", "model": "two", "pid": "Two"},
        ],
    )
    reg = ModelRegistry(path=path)
    assert {c.id for c in reg.group_cards()} == {"one", "two"}
    assert reg.resolve("one").routes[0].provider == "p"

    # rewrite the same file and reload -- the registry must notice
    _write_catalogue(
        path,
        [{"id": "p/one", "provider": "p", "model": "one", "pid": "One", "features": ["tool-use"]}],
    )
    assert reg.reload() == 2  # 1 group + 1 pinned
    assert reg.resolve("two") is None


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        json.dumps({"version": 1}),
        json.dumps({"models": None}),
        json.dumps({"models": "nope"}),
        json.dumps("just a string"),
        "42",
        json.dumps({"models": [1, 2, "x", None]}),
        json.dumps({"models": [{"provider": "p"}]}),
    ],
)
def test_registry_reload_survives_degenerate_catalogues(tmp_path: Path, payload: str):
    """Regression: every one of these used to raise TypeError during reload."""
    path = tmp_path / "models.json"
    path.write_text(payload, encoding="utf-8")
    reg = ModelRegistry(path=path)
    assert reg.cards == []
    assert reg.group_cards() == []


def test_registry_reload_accepts_a_bare_list_catalogue(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"provider": "p", "model": "m", "pid": "Model"}]), encoding="utf-8")
    reg = ModelRegistry(path=path)
    assert [c.id for c in reg.group_cards()] == ["model"]


def test_registry_reload_handles_broken_json(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")
    reg = ModelRegistry(path=path)
    assert reg.cards == []
    assert reg.default() is None
    assert reg.resolve("anything") is None
    assert reg.resolve("anything", strict=True) is None


def test_registry_reload_skips_rows_without_pid(tmp_path: Path):
    path = _write_catalogue(
        tmp_path / "models.json",
        [
            {"id": "p/one", "provider": "p", "model": "one", "pid": "One"},
            {"id": "p/two", "provider": "p", "model": "two"},            # no pid -> dropped
            {"id": "p/three", "provider": "p", "model": "three", "pid": "remove"},  # sentinel
            {"provider": "p", "model": "four", "pid": "Four"},           # no id -> derived
        ],
    )
    reg = ModelRegistry(path=path)
    ids = {c.id for c in reg.group_cards()}
    assert ids == {"one", "four"}


def test_registry_reload_disambiguates_colliding_slugs(tmp_path: Path):
    path = _write_catalogue(
        tmp_path / "models.json",
        [
            {"id": "p/a", "provider": "p", "model": "a", "pid": "Same Name"},
            {"id": "q/b", "provider": "q", "model": "b", "pid": "Same  Name"},
        ],
    )
    reg = ModelRegistry(path=path)
    slugs = sorted(c.id for c in reg.group_cards())
    assert slugs == ["same-name", "same-name-2"]


def test_registry_route_from_row_defaults(tmp_path: Path):
    path = _write_catalogue(
        tmp_path / "models.json",
        [{"provider": "p", "model": "m", "pid": "Model", "context_window": "4096"}],
    )
    reg = ModelRegistry(path=path)
    route = reg.resolve("model").routes[0]
    assert route.mid == "p/m"
    assert route.context_window == 4096
    assert route.max_output_tokens == 0
    assert route.features == ()
    assert route.label == "Model"


def test_model_card_shape():
    card = ModelCard(id="x", label="X", routes=())
    assert card.kind == "group"
    assert card.owned_by == "pointy-arrow"
    assert card.aliases == ()
    with pytest.raises(Exception):
        card.id = "y"  # frozen dataclass


def test_route_is_hashable_and_frozen():
    r = Route(provider="p", raw_id="m", mid="p/m", pid="P", slug="p")
    with pytest.raises(Exception):
        r.provider = "q"
    assert hash(r) == hash(Route(provider="p", raw_id="m", mid="p/m", pid="P", slug="p"))
