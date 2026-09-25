from __future__ import annotations
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
ROOT = Path(__file__).resolve().parent
MODELS_PATH = Path(os.getenv("PA_MODELS_PATH") or (ROOT / "models.json"))
FALLBACK_PATHS = (
    ROOT / "gratisfy_analysis" / "models.json",
    ROOT / "gratisfy_analysis" / "working_models.json",
)
PROVIDER_PREFERENCE: Dict[str, int] = {
    "voidai": 10,
    "nvidia-nim": 14,
    "groq": 16,
    "cloudflare": 18,
    "cerebras": 20,
    "google-ai-studio": 22,
    "mistral-codestral": 26,
    "evolvex": 28,
    "atessa": 30,
    "llmgateway": 32,
    "logfare": 34,
    "naga": 36,
    "openrouter": 38,
    "pollinations": 40,
    "aqua": 44,
    "meganova": 46,
    "zai": 48,
    "unorouter": 50,
    "routmy": 60,
    "ai-horde": 80,
}
DEFAULT_PREFERENCE = 50
ROUTE_COOLDOWN_S = 90.0
ROUTE_FAILURE_LIMIT = 3
def _slugify(pid: str) -> str:
    s = pid.strip().lower()
    s = re.sub(r"[@+_]", "-", s)
    s = re.sub(r"[^a-z0-9.\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s or "model"
@dataclass(frozen=True)
class Route:
    provider: str
    raw_id: str
    mid: str
    pid: str
    slug: str
    context_window: int = 0
    max_output_tokens: int = 0
    features: Tuple[str, ...] = ()
    input_modalities: Tuple[str, ...] = ()
    output_modalities: Tuple[str, ...] = ()
    supported_parameters: Tuple[str, ...] = ()
    label: str = ""
    @property
    def preference(self) -> int:
        return PROVIDER_PREFERENCE.get(self.provider, DEFAULT_PREFERENCE)
@dataclass(frozen=True)
class ModelCard:
    id: str
    label: str
    routes: Tuple[Route, ...]
    context_window: int = 0
    owned_by: str = "pointy-arrow"
    kind: str = "group"
    aliases: Tuple[str, ...] = ()
@dataclass
class RouteHealth:
    failures: int = 0
    cooldown_until: float = 0.0
    def available(self, now: float) -> bool:
        return now >= self.cooldown_until
    def report(self, ok: bool, hard: bool = False) -> None:
        if ok:
            self.failures = 0
            self.cooldown_until = 0.0
            return
        self.failures += 1
        if hard or self.failures >= ROUTE_FAILURE_LIMIT:
            self.cooldown_until = time.time() + ROUTE_COOLDOWN_S
            self.failures = 0
def _find_models_file() -> Optional[Path]:
    for p in (MODELS_PATH, *FALLBACK_PATHS):
        if p.is_file():
            return p
    return None
def _as_tuple(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return ()


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _route_from_row(row: Dict[str, Any]) -> Optional[Route]:
    provider = str(row.get("provider") or "").strip()
    raw_id = str(row.get("model") or row.get("id") or "").strip()
    mid = str(row.get("id") or f"{provider}/{raw_id}").strip()
    pid = str(row.get("pid") or "").strip()
    if not provider or not raw_id or not pid or pid.lower() == "remove":
        return None
    return Route(
        provider=provider,
        raw_id=raw_id,
        mid=mid,
        pid=pid,
        slug="",
        context_window=_as_int(row.get("context_window")),
        max_output_tokens=_as_int(row.get("max_output_tokens")),
        features=_as_tuple(row.get("features")),
        input_modalities=_as_tuple(row.get("input_modalities")),
        output_modalities=_as_tuple(row.get("output_modalities")),
        supported_parameters=_as_tuple(row.get("supported_parameters")),
        label=str(row.get("label") or pid),
    )
class ModelRegistry:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _find_models_file()
        self._lock = threading.Lock()
        self._cards: List[ModelCard] = []
        self._by_key: Dict[str, ModelCard] = {}
        self._route_health: Dict[str, RouteHealth] = {}
        self._mtime: float = 0.0
        self.reload()
    def reload(self) -> int:
        path = self.path or _find_models_file()
        rows: List[Dict[str, Any]] = []
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if isinstance(data, dict):
                data = data.get("models")
            # A catalogue that is valid JSON but carries no "models" list
            # (truncated write, hand-edited file, wrong shape) must degrade to
            # "no models" instead of taking the whole process down.
            if isinstance(data, list):
                rows = [row for row in data if isinstance(row, dict)]
        routes: List[Route] = []
        for row in rows:
            r = _route_from_row(row)
            if r:
                routes.append(r)
        groups: Dict[str, List[Route]] = {}
        for r in routes:
            groups.setdefault(r.pid, []).append(r)
        used_slugs: Dict[str, str] = {}
        cards: List[ModelCard] = []
        slugged_routes: List[Route] = []
        for pid in sorted(groups, key=str.lower):
            base = _slugify(pid)
            slug = base
            if slug in used_slugs and used_slugs[slug] != pid:
                i = 2
                while f"{base}-{i}" in used_slugs:
                    i += 1
                slug = f"{base}-{i}"
            used_slugs[slug] = pid
            group_routes = tuple(
                sorted(groups[pid], key=lambda r: (r.preference, r.provider, r.raw_id))
            )
            group_routes = tuple(
                Route(**{**r.__dict__, "slug": slug}) for r in group_routes
            )
            slugged_routes.extend(group_routes)
            ctx = max((r.context_window for r in group_routes), default=0)
            cards.append(
                ModelCard(
                    id=slug,
                    label=pid,
                    routes=group_routes,
                    context_window=ctx,
                    kind="group",
                    aliases=(pid,),
                )
            )
        for r in sorted(slugged_routes, key=lambda r: (r.provider.lower(), r.raw_id.lower())):
            cards.append(
                ModelCard(
                    id=r.mid,
                    label=f"{r.pid} @ {r.provider}",
                    routes=(r,),
                    context_window=r.context_window,
                    kind="pinned",
                    aliases=(),
                )
            )
        by_key: Dict[str, ModelCard] = {}
        for card in cards:
            by_key[card.id.lower()] = card
            by_key[card.label.lower()] = card
            for a in card.aliases:
                by_key[a.lower()] = card
        with self._lock:
            self._cards = cards
            self._by_key = by_key
            self._mtime = path.stat().st_mtime if path and path.is_file() else 0.0
        return len(cards)
    @property
    def cards(self) -> List[ModelCard]:
        with self._lock:
            return list(self._cards)
    def group_cards(self) -> List[ModelCard]:
        return [c for c in self.cards if c.kind == "group"]
    def default(self) -> Optional[ModelCard]:
        cards = self.cards
        if not cards:
            return None
        for preferred in ("glm-5-2", "kimi-k3", "grok-4-5"):
            card = self.resolve(preferred, strict=True)
            if card:
                return card
        return cards[0]
    def resolve(self, model_id: Optional[str], strict: bool = False) -> Optional[ModelCard]:
        if not model_id:
            return None if strict else self.default()
        key = model_id.strip().lower()
        if key in {"", "auto", "default", "router/default"}:
            return None if strict else self.default()
        with self._lock:
            card = self._by_key.get(key)
        if card:
            return card
        norm = _slugify(key)
        with self._lock:
            for card in self._cards:
                if card.id == norm:
                    return card
                if _slugify(card.label) == norm:
                    return card
        return None
    def route_health(self, mid: str) -> RouteHealth:
        with self._lock:
            health = self._route_health.get(mid)
            if health is None:
                health = RouteHealth()
                self._route_health[mid] = health
            return health
    def report_route(self, mid: str, ok: bool, hard: bool = False) -> None:
        self.route_health(mid).report(ok, hard=hard)
    def ordered_routes(self, card: ModelCard, prefer_tools: bool = False) -> List[Route]:
        now = time.time()
        healthy = [r for r in card.routes if self.route_health(r.mid).available(now)]
        if not healthy:
            healthy = list(card.routes)
        if prefer_tools:
            healthy.sort(key=lambda r: (0 if "tool-use" in r.features else 1, r.preference, r.provider))
        return healthy
REGISTRY = ModelRegistry()