from __future__ import annotations
import itertools
import json
import os
import random as _random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import requests
from pa_credentials import (
    Credential,
    CredentialFile,
    DEFAULT_OUTPUT,
    DEFAULT_USER_AGENT,
    ensure_fresh,
    harvest_one,
    _HttpError as CredentialHttpError,
)
from pa_models import ModelCard, ModelRegistry, Route, REGISTRY
AUTO_REFRESH_TOKEN = os.getenv("PA_AUTO_REFRESH", "1") not in {"0", "false", "False", ""}
AUTO_HARVEST = os.getenv("PA_AUTO_HARVEST", "1") not in {"0", "false", "False", ""}
CHAT_ENDPOINT = "https://gratisfy.xyz/api/chat"
DEFAULT_WEBSITE_ROUTE = "chat"
MIN_WORKING = 1
TARGET_WORKING = 3
MAX_TOTAL = 6
MAX_FAILURES = 3
COOLDOWN_BASE = 15.0
COOLDOWN_MAX = 300.0
HTTP_TIMEOUT = (10, 45)
MAX_CRED_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 20.0
MAX_MESSAGES = 40
MESSAGE_TRIM_TARGET = 30
class _StreamError(RuntimeError):
    pass
class _RouteRejected(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status
def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
    return "".join(parts)
def to_gratisfy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        content = msg.get("content")
        if role == "assistant" and msg.get("tool_calls"):
            text = _flatten_content(content)
            if text:
                out.append({"role": "assistant", "content": text})
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append(f"{fn.get('name', 'tool')}({fn.get('arguments', '')})")
            if calls:
                out.append({"role": "assistant", "content": "[called " + "; ".join(calls) + "]"})
            continue
        if role == "tool":
            name = msg.get("name") or "tool"
            out.append({"role": "user", "content": f"[{name} result] {_flatten_content(content)}"})
            continue
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif part.get("type") == "image_url":
                    iu = part.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
            if parts:
                if len(parts) == 1 and parts[0].get("type") == "text":
                    out.append({"role": role, "content": parts[0]["text"]})
                else:
                    out.append({"role": role, "content": parts})
                continue
        out.append({"role": role, "content": _flatten_content(content)})
    return out
class _GratisfyClient:
    def __init__(self, credential: Credential, timeout: Tuple[int, int] = HTTP_TIMEOUT) -> None:
        self.credential = credential
        self.timeout = timeout
        self.session = requests.Session()
    def _headers(self) -> Dict[str, str]:
        ensure_fresh(self.credential)
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Origin": "https://gratisfy.xyz",
            "Referer": "https://gratisfy.xyz/chat",
            "User-Agent": self.credential.user_agent or DEFAULT_USER_AGENT,
            "Authorization": self.credential.bearer_header(),
            "X-Website-Route": DEFAULT_WEBSITE_ROUTE,
            "X-Telemetry-Session": self.credential.id,
        }
    @staticmethod
    def _build_payload(
        messages: List[Dict[str, Any]],
        route: Route,
        *,
        tools: Optional[List[Any]],
        tool_choice: Optional[Any],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        reasoning_effort: Optional[str],
        response_format: Optional[Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": route.raw_id,
            "provider": route.provider,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format
        return payload
    def stream(
        self,
        messages: List[Dict[str, Any]],
        route: Route,
        *,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Any] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        payload = self._build_payload(
            messages, route,
            tools=tools, tool_choice=tool_choice,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            reasoning_effort=reasoning_effort, response_format=response_format,
        )
        with self.session.post(
            CHAT_ENDPOINT, headers=self._headers(), json=payload,
            stream=True, timeout=self.timeout,
        ) as resp:
            if resp.status_code != 200:
                self._raise_for_rejection(resp)
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                stripped = line.strip()
                if stripped.startswith(":"):
                    continue
                if stripped.startswith("data:"):
                    stripped = stripped[5:].strip()
                if not stripped:
                    continue
                if stripped == "[DONE]":
                    yield {"type": "done"}
                    return
                if not stripped.startswith("{"):
                    continue
                try:
                    obj = json.loads(stripped)
                except Exception:
                    continue
                for ev in self._events_from_obj(obj):
                    yield ev
    @staticmethod
    def _raise_for_rejection(resp: requests.Response) -> None:
        status = resp.status_code
        try:
            body = resp.text or ""
        except Exception:
            body = ""
        low = body.lower()
        if status == 401 and "website_auth_required" in low:
            resp.raise_for_status()
        if status in (401, 403) and "authentication is required" in low:
            resp.raise_for_status()
        if status == 429:
            raise requests.HTTPError(f"rate limited: {body[:200]}", response=resp)
        if 500 <= status < 600:
            raise requests.HTTPError(f"upstream {status}: {body[:200]}", response=resp)
        raise _RouteRejected(f"status {status}: {body[:300]}", status=status)
    def _events_from_obj(self, obj: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        err = obj.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            yield {"type": "error", "message": msg or "unknown upstream error"}
            return
        usage = obj.get("usage")
        if usage:
            yield {"type": "usage", "usage": usage}
        choices = obj.get("choices") or []
        if not choices:
            return
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            if _is_gratisfy_error_text(content):
                yield {"type": "error", "message": content.strip()[:300]}
            else:
                yield {"type": "content", "text": content}
        elif isinstance(content, list):
            text_parts: List[str] = []
            think_parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "thinking":
                    for item in part.get("thinking") or []:
                        if isinstance(item, str):
                            think_parts.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("text"), str):
                            think_parts.append(item["text"])
                elif isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if think_parts:
                yield {"type": "reasoning", "text": "".join(think_parts)}
            if text_parts:
                joined = "".join(text_parts)
                if _is_gratisfy_error_text(joined):
                    yield {"type": "error", "message": joined.strip()[:300]}
                else:
                    yield {"type": "content", "text": joined}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "reasoning", "text": reasoning}
        details = delta.get("reasoning_details")
        if isinstance(details, list):
            texts: List[str] = []
            for item in details:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                yield {"type": "reasoning", "text": "".join(texts)}
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            yield {
                "type": "tool_call_delta",
                "index": int(tc.get("index", 0) or 0),
                "id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": fn.get("arguments") or "",
            }
        finish = choice.get("finish_reason")
        if finish:
            if str(finish).lower() == "error":
                yield {"type": "error", "message": "upstream stream ended with finish_reason=error"}
            else:
                yield {"type": "finish", "finish_reason": str(finish)}
@dataclass
class CredentialState:
    credential: Credential
    failures: int = 0
    successes: int = 0
    depleted: bool = False
    cooldown_until: float = 0.0
    last_used: float = 0.0
    last_error: Optional[str] = None
    def available(self, now: float) -> bool:
        return (not self.depleted) and now >= self.cooldown_until
class NoCredentialsError(RuntimeError):
    pass
class AllCredentialsBusyError(RuntimeError):
    pass
class CredentialPool:
    def __init__(self, path: Path = DEFAULT_OUTPUT) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._states: List[CredentialState] = []
        self._cycle = itertools.cycle([])
        self._harvesting = False
        self.reload()
    def reload(self) -> int:
        creds = CredentialFile(self.path).load()
        with self._lock:
            existing = {s.credential.id: s for s in self._states}
            states: List[CredentialState] = []
            for cred in creds:
                if cred.id in existing:
                    state = existing[cred.id]
                    state.credential = cred
                    states.append(state)
                else:
                    states.append(CredentialState(credential=cred))
            self._states = states
            self._rebuild_cycle()
        return len(creds)
    def _rebuild_cycle(self) -> None:
        self._cycle = itertools.cycle(range(len(self._states))) if self._states else itertools.cycle([])
    def _save(self) -> None:
        CredentialFile(self.path).save([s.credential for s in self._states if not s.depleted])
    def total(self) -> int:
        with self._lock:
            return len(self._states)
    def working(self) -> int:
        with self._lock:
            return sum(1 for s in self._states if not s.depleted)
    def prune(self) -> int:
        with self._lock:
            before = len(self._states)
            self._states = [s for s in self._states if not s.depleted]
            self._rebuild_cycle()
            self._save()
            return before - len(self._states)
    def stats(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                "total": len(self._states),
                "working": sum(1 for s in self._states if not s.depleted),
                "available": sum(1 for s in self._states if s.available(now)),
                "harvesting": self._harvesting,
                "credentials": [
                    {
                        "id": s.credential.id,
                        "email": s.credential.email,
                        "successes": s.successes,
                        "failures": s.failures,
                        "depleted": s.depleted,
                        "cooldown_remaining": max(0.0, round(s.cooldown_until - now, 1)),
                        "last_error": s.last_error,
                    }
                    for s in self._states
                ],
            }
    def acquire(self, exclude_ids: Optional[set] = None) -> Optional[CredentialState]:
        now = time.time()
        exclude_ids = exclude_ids or set()
        with self._lock:
            n = len(self._states)
            for _ in range(n):
                idx = next(self._cycle)
                state = self._states[idx]
                if not state.available(now) or state.credential.id in exclude_ids:
                    continue
                state.last_used = now
                picked = state
                break
            else:
                picked = None
        if self.working() <= MIN_WORKING:
            self.maybe_harvest()
        return picked
    def report_result(self, state: CredentialState, ok: bool, depleted: bool = False, error: Optional[str] = None) -> None:
        changed = False
        with self._lock:
            if ok:
                state.successes += 1
                state.failures = 0
                state.cooldown_until = 0.0
                state.last_error = None
            else:
                state.failures += 1
                state.last_error = error
                if depleted or state.failures >= MAX_FAILURES:
                    state.depleted = True
                    changed = True
                else:
                    backoff = min(COOLDOWN_BASE * (2 ** (state.failures - 1)), COOLDOWN_MAX)
                    state.cooldown_until = time.time() + backoff
            if changed:
                self._states = [s for s in self._states if not s.depleted]
                self._rebuild_cycle()
                self._save()
        if self.working() <= MIN_WORKING:
            self.maybe_harvest()
    def save_tokens(self, state: CredentialState) -> None:
        self._save()
    def maybe_harvest(self) -> None:
        if not AUTO_HARVEST:
            return
        with self._lock:
            if self._harvesting:
                return
            working = sum(1 for s in self._states if not s.depleted)
            if working >= TARGET_WORKING or len(self._states) >= MAX_TOTAL:
                return
            self._harvesting = True
        threading.Thread(target=self._harvest_loop, daemon=True).start()
    def _harvest_loop(self) -> None:
        try:
            while self.working() < TARGET_WORKING and self.total() < MAX_TOTAL:
                try:
                    cred = harvest_one()
                except Exception:
                    break
                if not cred.is_valid():
                    continue
                with self._lock:
                    self._states.append(CredentialState(credential=cred))
                    self._rebuild_cycle()
                    self._save()
        finally:
            with self._lock:
                self._harvesting = False
    def ensure_ready(self, block: bool = True) -> None:
        self.prune()
        if AUTO_REFRESH_TOKEN:
            with self._lock:
                states = list(self._states)
            for state in states:
                cred = state.credential
                if not cred.needs_refresh():
                    continue
                try:
                    ensure_fresh(cred)
                    self._save()
                except Exception:
                    with self._lock:
                        state.depleted = True
                        state.last_error = "refresh failed"
            self.prune()
        if self.working() <= MIN_WORKING:
            if block and AUTO_HARVEST:
                attempts = 0
                while self.working() <= MIN_WORKING and self.total() < MAX_TOTAL and attempts < MAX_TOTAL:
                    attempts += 1
                    try:
                        cred = harvest_one()
                    except Exception as exc:
                        print(f"harvest error: {exc}")
                        break
                    if cred.is_valid():
                        with self._lock:
                            self._states.append(CredentialState(credential=cred))
                            self._rebuild_cycle()
                            self._save()
            self.maybe_harvest()
class PARouter:
    def __init__(self, pool: Optional[CredentialPool] = None, registry: ModelRegistry = REGISTRY) -> None:
        self.pool = pool or CredentialPool()
        self.registry = registry
    def ensure_ready(self, block: bool = True) -> None:
        self.pool.ensure_ready(block=block)
    def stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        *,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Any] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        card = self.registry.resolve(model)
        if card is None:
            raise NoCredentialsError("model registry is empty — no curated models available")
        gratisfy_messages = to_gratisfy_messages(messages)
        if len(gratisfy_messages) > MAX_MESSAGES:
            system_msgs = [m for m in gratisfy_messages if m.get("role") == "system"]
            rest = [m for m in gratisfy_messages if m.get("role") != "system"]
            gratisfy_messages = system_msgs + rest[-MESSAGE_TRIM_TARGET:]
        if self.pool.total() == 0:
            self.pool.ensure_ready(block=True)
        if self.pool.total() == 0:
            raise NoCredentialsError(
                "No Gratisfy credentials loaded. Add sessions to credentials.json "
                "or let auto-harvest create one, then POST /admin/reload."
            )
        last_error: Optional[Exception] = None
        for route in self.registry.ordered_routes(card):
            emitted = False
            for _attempt in range(MAX_CRED_ATTEMPTS):
                state = self.pool.acquire()
                if state is None:
                    if any(not s.depleted for s in self.pool._states):
                        raise AllCredentialsBusyError("All working credentials are cooling down; try again shortly.")
                    raise NoCredentialsError("No credentials available.")
                client = _GratisfyClient(state.credential)
                try:
                    for ev in client.stream(
                        gratisfy_messages, route,
                        tools=tools, tool_choice=tool_choice,
                        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort, response_format=response_format,
                    ):
                        etype = ev.get("type")
                        if etype == "error":
                            msg = str(ev.get("message", "upstream error"))
                            if _looks_route_scoped(msg):
                                raise _RouteRejected(msg)
                            raise _StreamError(msg)
                        if not emitted:
                            emitted = True
                            yield {
                                "type": "route",
                                "credential_id": state.credential.id,
                                "mid": route.mid,
                                "pid": route.pid,
                                "provider": route.provider,
                                "model": card.id,
                            }
                        yield ev
                    self.pool.report_result(state, ok=True)
                    self.pool.save_tokens(state)
                    self.registry.report_route(route.mid, ok=True)
                    return
                except _RouteRejected as exc:
                    last_error = exc
                    self.registry.report_route(route.mid, ok=False)
                    break
                except CredentialHttpError as exc:
                    depleted = exc.status in (401, 403)
                    self.pool.report_result(state, ok=False, depleted=depleted, error=f"cred http {exc.status}")
                    last_error = exc
                    if emitted:
                        raise
                    _sleep_jitter(_attempt)
                    continue
                except requests.HTTPError as exc:
                    code = exc.response.status_code if exc.response is not None else None
                    depleted = code in (401, 403)
                    self.pool.report_result(state, ok=False, depleted=depleted, error=f"http {code}")
                    last_error = exc
                    if emitted:
                        raise
                    _sleep_jitter(_attempt, exc.response.headers.get("Retry-After") if exc.response is not None else None)
                    continue
                except _StreamError as exc:
                    self.pool.report_result(state, ok=False, depleted=False, error=str(exc))
                    last_error = exc
                    if emitted:
                        raise
                    _sleep_jitter(_attempt)
                    continue
                except Exception as exc:
                    self.pool.report_result(state, ok=False, depleted=False, error=str(exc))
                    last_error = exc
                    if emitted:
                        raise
                    _sleep_jitter(_attempt)
                    continue
        raise RuntimeError(f"All routes for '{card.id}' failed. Last error: {last_error}")
    def collect(self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_acc: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        usage: Optional[Dict[str, Any]] = None
        finish_reason = "stop"
        route_info: Optional[Dict[str, Any]] = None
        for ev in self.stream(messages, model=model, **kwargs):
            etype = ev.get("type")
            if etype == "route":
                route_info = ev
            elif etype == "content":
                text_parts.append(ev.get("text", ""))
            elif etype == "reasoning":
                reasoning_parts.append(ev.get("text", ""))
            elif etype == "tool_call_delta":
                index = ev.get("index", 0)
                if index not in tool_acc:
                    tool_acc[index] = {
                        "id": ev.get("id") or f"call_{int(time.time()*1000):x}{len(order):02x}",
                        "name": ev.get("name") or "",
                        "arguments": "",
                    }
                    order.append(index)
                if ev.get("id"):
                    tool_acc[index]["id"] = ev["id"]
                if ev.get("name"):
                    tool_acc[index]["name"] = ev["name"]
                tool_acc[index]["arguments"] += ev.get("arguments") or ""
            elif etype == "usage":
                usage = ev.get("usage")
            elif etype == "finish":
                finish_reason = ev.get("finish_reason", finish_reason)
        tool_calls = [
            {
                "id": tool_acc[i]["id"],
                "type": "function",
                "function": {"name": tool_acc[i]["name"], "arguments": tool_acc[i]["arguments"]},
            }
            for i in order
        ]
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        return {
            "text": "".join(text_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": tool_calls,
            "usage": usage,
            "route": route_info,
            "finish_reason": finish_reason,
        }
def _sleep_jitter(attempt: int, retry_after: Optional[str] = None) -> None:
    delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt)) + _random.random()
    if retry_after:
        try:
            requested = float(retry_after)
            delay = min(max(delay, requested), 15.0)
        except ValueError:
            pass
    time.sleep(delay)
def _looks_route_scoped(message: str) -> bool:
    low = message.lower()
    markers = (
        "payment", "402", "insufficient balance", "credit balance",
        "not found", "404", "does not exist",
        "unauthorized", "401", "403", "key limit",
        "billing", "entitlement",
        "an error occurred", "finish_reason=error",
    )
    return any(m in low for m in markers)
def _is_gratisfy_error_text(text: str) -> bool:
    return "[an error occurred" in text.strip().lower()