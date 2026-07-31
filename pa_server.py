from __future__ import annotations
import functools
import json
import queue as _queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pa_models import ModelCard, REGISTRY
from pa_router import (
    AllCredentialsBusyError,
    CredentialPool,
    NoCredentialsError,
    PARouter,
)
SERVER_NAME = "pointy-arrow-api"
KEEPALIVE_SECONDS = 4.0
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    reasoning_effort: Optional[str] = None
    response_format: Optional[Any] = None
    max_completion_tokens: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None
def _now() -> int:
    return int(time.time())
def _error_body(message: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "server_error", "code": code}},
    )
def _model_card(card: ModelCard) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": card.id,
        "object": "model",
        "created": _now(),
        "owned_by": card.owned_by,
    }
    if card.context_window:
        out["context_window"] = card.context_window
    out["label"] = card.label
    if card.kind == "group":
        out["providers"] = [r.provider for r in card.routes]
    else:
        out["providers"] = [card.routes[0].provider]
        out["group"] = card.routes[0].slug
    return out
def _flatten_content(content: Union[str, List[Dict[str, Any]], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)
def _messages_to_dicts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.name:
            entry["name"] = msg.name
        if msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        out.append(entry)
    return out
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
def _normalize_usage(usage: Optional[Dict[str, Any]], prompt_text: str, completion_text: str) -> Dict[str, int]:
    usage = usage or {}
    prompt = (
        usage.get("prompt_tokens")
        or usage.get("promptTokens")
        or usage.get("input_tokens")
        or usage.get("inputTokens")
        or _estimate_tokens(prompt_text)
    )
    completion = (
        usage.get("completion_tokens")
        or usage.get("completionTokens")
        or usage.get("output_tokens")
        or usage.get("outputTokens")
        or usage.get("candidatesTokenCount")
        or _estimate_tokens(completion_text)
    )
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(prompt) + int(completion),
    }
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    router = PARouter()
    app.state.router = router
    print("refreshing pool...")
    try:
        await anyio.to_thread.run_sync(lambda: router.ensure_ready(block=True))
    except Exception as e:
        print(f"fetch error: {e}")
    print(f"{router.pool.working()}/{router.pool.total()} in pool, {len(REGISTRY.group_cards())} models")
    yield
app = FastAPI(title=SERVER_NAME, version="2.0.0", lifespan=lifespan)
def get_router(request: Request) -> PARouter:
    return request.app.state.router
@app.get("/v1/models")
@app.get("/models")
async def list_models() -> Dict[str, Any]:
    cards = REGISTRY.group_cards() + [c for c in REGISTRY.cards if c.kind == "pinned"]
    return {"object": "list", "data": [_model_card(c) for c in cards]}
@app.get("/v1/models/{model_id:path}")
@app.get("/models/{model_id:path}")
async def retrieve_model(model_id: str) -> Dict[str, Any]:
    card = REGISTRY.resolve(model_id, strict=True)
    if card is None:
        return JSONResponse(status_code=404, content={"error": {"message": f"The model '{model_id}' does not exist", "type": "invalid_request_error", "code": "model_not_found"}})
    return _model_card(card)
@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    router = get_router(request)
    return {
        "status": "ok",
        "server": SERVER_NAME,
        "models": {"groups": len(REGISTRY.group_cards()), "pinned": sum(1 for c in REGISTRY.cards if c.kind == "pinned")},
        "credentials": router.pool.stats(),
    }
@app.post("/admin/reload")
async def reload_state(request: Request) -> Dict[str, Any]:
    router = get_router(request)
    cred_count = router.pool.reload()
    card_count = REGISTRY.reload()
    return {
        "credentials_reloaded": cred_count,
        "models_reloaded": card_count,
        "credentials": router.pool.stats(),
    }
def _sse_chunk(cid: str, created: int, model_id: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
async def _stream_response(router: PARouter, messages: List[Dict[str, Any]], card: ModelCard, body: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = _now()
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    prompt_text = "".join(_flatten_content(m.content) for m in body.messages)
    completion_so_far: List[str] = []
    started = False
    tool_acc: Dict[int, Dict[str, Any]] = {}
    tool_order: List[int] = []
    tool_announced: Dict[int, bool] = {}
    gen = router.stream(
        messages, model=card.id,
        tools=body.tools, tool_choice=body.tool_choice,
        temperature=body.temperature, top_p=body.top_p,
        max_tokens=body.max_tokens or body.max_completion_tokens,
        reasoning_effort=body.reasoning_effort, response_format=body.response_format,
    )
    events: "_queue.Queue[Any]" = _queue.Queue()
    _sentinel = object()
    def _pump() -> None:
        try:
            for item in gen:
                events.put(("ev", item))
        except BaseException as exc:
            events.put(("exc", exc))
        finally:
            events.put(_sentinel)
    threading.Thread(target=_pump, daemon=True).start()
    try:
        while True:
            try:
                item = await anyio.to_thread.run_sync(functools.partial(events.get, timeout=KEEPALIVE_SECONDS))
            except _queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is _sentinel:
                break
            kind, ev = item
            if kind == "exc":
                raise ev
            etype = ev.get("type")
            if etype == "content":
                text = ev.get("text", "")
                if not text:
                    continue
                if not started:
                    started = True
                    yield _sse_chunk(cid, created, card.id, {"role": "assistant", "content": ""})
                completion_so_far.append(text)
                yield _sse_chunk(cid, created, card.id, {"content": text})
            elif etype == "reasoning":
                text = ev.get("text", "")
                if not text:
                    continue
                if not started:
                    started = True
                    yield _sse_chunk(cid, created, card.id, {"role": "assistant", "content": ""})
                yield _sse_chunk(cid, created, card.id, {"reasoning_content": text})
            elif etype == "tool_call_delta":
                if not started:
                    started = True
                    yield _sse_chunk(cid, created, card.id, {"role": "assistant", "content": ""})
                index = ev.get("index", 0)
                if index not in tool_acc:
                    tool_acc[index] = {"id": ev.get("id") or f"call_{uuid.uuid4().hex[:24]}", "name": ev.get("name") or ""}
                    tool_order.append(index)
                if ev.get("id"):
                    tool_acc[index]["id"] = ev["id"]
                if ev.get("name"):
                    tool_acc[index]["name"] = ev["name"]
                out_index = tool_order.index(index)
                if not tool_announced.get(index):
                    tool_announced[index] = True
                    delta = {
                        "tool_calls": [{
                            "index": out_index,
                            "id": tool_acc[index]["id"],
                            "type": "function",
                            "function": {"name": tool_acc[index]["name"], "arguments": ev.get("arguments") or ""},
                        }]
                    }
                else:
                    delta = {"tool_calls": [{"index": out_index, "function": {"arguments": ev.get("arguments") or ""}}]}
                yield _sse_chunk(cid, created, card.id, delta)
                finish_reason = "tool_calls"
            elif etype == "usage":
                usage = ev.get("usage")
            elif etype == "finish":
                finish_reason = ev.get("finish_reason") or finish_reason
    except NoCredentialsError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'no_credentials'}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except AllCredentialsBusyError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'all_credentials_busy'}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
        if started:
            yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'upstream_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'upstream_error'}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    if not started:
        yield _sse_chunk(cid, created, card.id, {"role": "assistant", "content": ""})
    if tool_order:
        finish_reason = "tool_calls"
    yield _sse_chunk(cid, created, card.id, {}, finish_reason or "stop")
    if (body.stream_options or {}).get("include_usage"):
        final_usage = _normalize_usage(usage, prompt_text, "".join(completion_so_far))
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': card.id, 'choices': [], 'usage': final_usage}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    router = get_router(request)
    card = REGISTRY.resolve(body.model)
    if card is None:
        return _error_body("model registry is empty", "no_models", 503)
    messages = _messages_to_dicts(body.messages)
    if body.stream:
        return StreamingResponse(
            _stream_response(router, messages, card, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        result = await anyio.to_thread.run_sync(
            lambda: router.collect(
                messages, model=card.id,
                tools=body.tools, tool_choice=body.tool_choice,
                temperature=body.temperature, top_p=body.top_p,
                max_tokens=body.max_tokens or body.max_completion_tokens,
                reasoning_effort=body.reasoning_effort, response_format=body.response_format,
            )
        )
    except NoCredentialsError as exc:
        return _error_body(str(exc), "no_credentials", 503)
    except AllCredentialsBusyError as exc:
        return _error_body(str(exc), "all_credentials_busy", 429)
    except Exception as exc:
        return _error_body(str(exc), "upstream_error", 502)
    prompt_text = "".join(_flatten_content(m.content) for m in body.messages)
    completion_text = result["text"]
    usage = _normalize_usage(result.get("usage"), prompt_text, completion_text)
    message: Dict[str, Any] = {"role": "assistant", "content": completion_text or None}
    if result.get("reasoning"):
        message["reasoning_content"] = result["reasoning"]
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": _now(),
        "model": card.id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.get("finish_reason", "stop"),
        }],
        "usage": usage,
    })
def main() -> None:
    import uvicorn
    uvicorn.run("pa_server:app", host="127.0.0.1", port=8001, reload=False)
if __name__ == "__main__":
    main()