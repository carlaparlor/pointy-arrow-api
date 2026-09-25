"""End-to-end chat tests: real server process, real sockets, real SSE.

Each test drives the actual ``pa_server`` (started as its own uvicorn process)
against the reference upstream in ``tests/reference_upstream.py``.  The
reference upstream records every request it receives, so these tests assert on
what the router *actually* sent upstream -- payload shape, routing, retries and
credential handling -- rather than on any substituted object.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _client(api, timeout: float = 90.0) -> httpx.Client:
    return httpx.Client(base_url=api.base, timeout=timeout)


def _chat(api, payload: Dict[str, Any]) -> httpx.Response:
    return _client(api).post("/v1/chat/completions", json=payload)


def _hello(payload: Dict[str, Any]) -> Dict[str, Any]:
    body = {"model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]}
    body.update(payload)
    return body


def write_catalogue(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"version": 1, "models": rows}), encoding="utf-8")
    return path


def single_route_catalogue(path: Path, *, pid: str = "Solo Model", provider: str = "p",
                           model: str = "solo", features=("chat",)) -> Path:
    return write_catalogue(path, [{
        "id": f"{provider}/{model}", "provider": provider, "model": model, "pid": pid,
        "context_window": 4096, "features": list(features),
        "supported_parameters": ["messages", "stream", "tools", "tool_choice"],
    }])


def sse_frames(response: httpx.Response) -> List[Dict[str, Any]]:
    """Collect the JSON payloads from a finished SSE response."""
    frames = []
    for line in response.text.splitlines():
        line = line.strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            frames.append(json.loads(line[5:].strip()))
    return frames


def read_sse(response: httpx.Response) -> Dict[str, Any]:
    """Drain an open streaming response (must be called inside the `with`)."""
    raw = "".join(response.iter_text())
    frames = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            frames.append(json.loads(line[5:].strip()))
    content = "".join(f["choices"][0]["delta"].get("content", "") for f in frames
                      if f.get("choices"))
    return {"frames": frames, "content": content, "raw": raw}


# =========================================================================== #
# happy paths
# =========================================================================== #
def test_non_streaming_chat_completion(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"content": "Hello"}}, {"delta": {"content": ", world"}},
                   {"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }])
    r = _chat(api, _hello({}))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "kimi-k3"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello, world"}
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def test_the_upstream_receives_the_translated_payload(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    _chat(api, _hello({
        "temperature": 0.5, "top_p": 0.9, "max_tokens": 128,
        "reasoning_effort": "high", "response_format": {"type": "json_object"},
    }))
    reqs = _chat_requests(api)
    assert len(reqs) == 1
    payload = reqs[0]["body"]
    assert payload["model"] == "kimi-k3"
    assert payload["provider"] == "voidai"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 128
    assert payload["reasoning_effort"] == "high"
    assert payload["response_format"] == {"type": "json_object"}
    headers = reqs[0]["headers"]
    assert headers["authorization"].startswith("Bearer access-token-")
    assert headers["x-website-route"] == "chat"
    assert headers["origin"] == "https://gratisfy.xyz"
    assert headers["content-type"] == "application/json"


def _chat_requests(api) -> List[Dict[str, Any]]:
    """Every chat request the reference upstream received for this server."""
    log = httpx.get(f"{api.upstream_base}/__log", timeout=10).json()["entries"]
    return [e for e in log if e["path"] == "/api/chat"]


def test_usage_is_estimated_when_upstream_reports_none(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "abcd" * 10}},
                                   {"finish_reason": "stop"}]}])
    r = _chat(api, _hello({}))
    usage = r.json()["usage"]
    # 2 chars prompt -> 1 token; 40 chars completion -> 10 tokens
    assert usage["prompt_tokens"] == 1
    assert usage["completion_tokens"] == 10
    assert usage["total_tokens"] == 11


@pytest.mark.parametrize("upstream_keys,expected", [
    ({"prompt_tokens": 11, "completion_tokens": 22}, (11, 22)),
    ({"promptTokens": 12, "completionTokens": 23}, (12, 23)),
    ({"input_tokens": 13, "output_tokens": 24}, (13, 24)),
    ({"inputTokens": 14, "outputTokens": 25}, (14, 25)),
])
def test_usage_key_aliases_are_normalised(api, scenario, upstream_keys, expected):
    scenario.set_chat([{"events": [{"delta": {"content": "x"}}, {"finish_reason": "stop"}],
                       "usage": upstream_keys}])
    usage = _chat(api, _hello({})).json()["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == expected
    assert usage["total_tokens"] == sum(expected)


def test_usage_with_gemini_style_keys(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "x"}}, {"finish_reason": "stop"}],
                       "usage": {"promptTokens": 5, "candidatesTokenCount": 6}}])
    usage = _chat(api, _hello({})).json()["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == 6


def test_streaming_chat_completion(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"content": "Hel"}}, {"delta": {"content": "lo"}},
                   {"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }])
    with _client(api).stream("POST", "/v1/chat/completions", json=_hello({"stream": True})) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        out = read_sse(r)
        assert r.headers["x-accel-buffering"] == "no"
    assert out["content"] == "Hello"
    assert out["raw"].endswith("data: [DONE]\n\n")
    first = out["frames"][0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    last = out["frames"][-1]
    assert last["choices"][0]["delta"] == {}
    assert last["choices"][0]["finish_reason"] == "stop"


def test_streaming_include_usage(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "hi"}}, {"finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 4, "completion_tokens": 1}}])
    with _client(api).stream("POST", "/v1/chat/completions",
                             json=_hello({"stream": True,
                                          "stream_options": {"include_usage": True}})) as r:
        out = read_sse(r)
    assert out["frames"][-1]["usage"] == {
        "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


def test_streaming_without_include_usage_omits_usage(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "hi"}}, {"finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 4, "completion_tokens": 1}}])
    with _client(api).stream("POST", "/v1/chat/completions", json=_hello({"stream": True})) as r:
        out = read_sse(r)
    assert "usage" not in out["frames"][-1]


def test_streaming_emits_keepalives_while_upstream_stalls(api, scenario):
    # the first frame arrives immediately, then the upstream goes quiet for 5s
    scenario.set_chat([{
        "events": [{"delta": {"content": "first"}},
                   {"delta": {"content": "second"}, "sleep_ms": 5200},
                   {"finish_reason": "stop"}],
    }])
    started = time.time()
    with _client(api, timeout=60).stream("POST", "/v1/chat/completions",
                                         json=_hello({"stream": True})) as r:
        out = read_sse(r)
    elapsed = time.time() - started
    assert elapsed >= 5.0, elapsed  # it really waited for the slow upstream
    assert ": keepalive" in out["raw"]
    assert out["content"] == "firstsecond"



def test_reasoning_is_reported_separately(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"reasoning_content": "let me think"}},
                   {"delta": {"content": "42"}}, {"finish_reason": "stop"}],
    }])
    r = _chat(api, _hello({}))
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "42"
    assert msg["reasoning_content"] == "let me think"


def test_reasoning_is_not_leaked_into_streamed_content(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"reasoning_content": "secret thoughts"}},
                   {"delta": {"content": "answer"}}, {"finish_reason": "stop"}],
    }])
    with _client(api).stream("POST", "/v1/chat/completions", json=_hello({"stream": True})) as r:
        out = read_sse(r)
    assert out["content"] == "answer"
    assert "secret thoughts" not in out["raw"]


def test_reasoning_details_are_flattened(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"reasoning_details": [{"text": "part1"}, {"text": "part2"}]}},
                   {"delta": {"content": "done"}}, {"finish_reason": "stop"}],
    }])
    r = _chat(api, _hello({}))
    assert r.json()["choices"][0]["message"]["reasoning_content"] == "part1part2"


def test_array_style_content_parts_are_flattened(api, scenario):
    scenario.set_chat([{
        "events": [{"delta": {"content": [{"type": "text", "text": "a"},
                                          {"type": "text", "text": "b"}]}},
                   {"finish_reason": "stop"}],
    }])
    assert _chat(api, _hello({})).json()["choices"][0]["message"]["content"] == "ab"


# =========================================================================== #
# tool calling
# =========================================================================== #
def test_native_tool_calls_are_forwarded(api, scenario):
    scenario.set_chat([{
        "events": [
            {"delta": {"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                       "function": {"name": "get_weather",
                                                    "arguments": '{"ci'}}]}},
            {"delta": {"tool_calls": [{"index": 0,
                                       "function": {"arguments": 'ty":"Oslo"}'}}]}},
            {"finish_reason": "tool_calls"},
        ],
    }])
    r = _chat(api, _hello({"tools": [WEATHER_TOOL], "tool_choice": "auto"}))
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["tool_calls"] == [{
        "id": "call_abc", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
    }]
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"

    payload = _chat_requests(api)[0]["body"]
    assert payload["tools"] == [WEATHER_TOOL]
    assert payload["tool_choice"] == "auto"


def test_streamed_tool_calls(api, scenario):
    scenario.set_chat([{
        "events": [
            {"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                       "function": {"name": "get_weather",
                                                    "arguments": '{"city":'}}]}},
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Oslo"}'}}]}},
            {"finish_reason": "tool_calls"},
        ],
    }])
    with _client(api).stream("POST", "/v1/chat/completions",
                             json=_hello({"stream": True, "tools": [WEATHER_TOOL]})) as r:
        out = read_sse(r)
    tool_frames = [f for f in out["frames"]
                   if f["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_frames) == 2
    first = tool_frames[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first["index"] == 0
    assert first["id"] == "call_1"
    assert first["function"]["name"] == "get_weather"
    assert out["frames"][-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_parallel_native_tool_calls(api, scenario):
    scenario.set_chat([{
        "events": [
            {"delta": {"tool_calls": [{"index": 0, "id": "c0",
                                       "function": {"name": "a", "arguments": "{}"}}]}},
            {"delta": {"tool_calls": [{"index": 1, "id": "c1",
                                       "function": {"name": "b", "arguments": "{}"}}]}},
            {"finish_reason": "tool_calls"},
        ],
    }])
    r = _chat(api, _hello({"tools": [WEATHER_TOOL]}))
    calls = r.json()["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert [c["id"] for c in calls] == ["c0", "c1"]


def test_tool_choice_none_strips_tools_from_the_upstream_payload(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "no tools"}},
                                   {"finish_reason": "stop"}]}])
    r = _chat(api, _hello({"tools": [WEATHER_TOOL], "tool_choice": "none"}))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "no tools"
    payload = _chat_requests(api)[0]["body"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_emulated_tool_calls_when_the_route_lacks_tool_use(api_factory, upstream, tmp_path,
                                                           scenario):
    """A route without the tool-use feature falls back to JSON emulation."""
    models = single_route_catalogue(tmp_path / "models.json", features=("chat",))
    srv = api_factory(models_path=models)
    try:
        upstream.scenario.set_chat([
            {"events": [{"delta": {"content": 'Let me check. {"tool_call": '
                                               '{"name": "get_weather", '
                                               '"arguments": {"city": "Oslo"}}}'}},
                        {"finish_reason": "stop"}]},
        ])
        r = _chat(srv, {"model": "solo-model",
                        "messages": [{"role": "user", "content": "weather in Oslo?"}],
                        "tools": [WEATHER_TOOL]})
        assert r.status_code == 200, r.text
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "Let me check."
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}
        assert r.json()["choices"][0]["finish_reason"] == "tool_calls"

        payload = _chat_requests(srv)[0]["body"]
        assert "tools" not in payload  # emulation never forwards the tool schema
        assert payload["messages"][-1]["role"] == "system"
        assert "[TOOL CALLING MODE]" in payload["messages"][-1]["content"]
        assert "get_weather" in payload["messages"][-1]["content"]
    finally:
        srv.stop()


def test_emulated_parallel_tool_calls(api_factory, upstream, tmp_path):
    models = single_route_catalogue(tmp_path / "models.json", features=("chat",))
    srv = api_factory(models_path=models)
    try:
        upstream.scenario.set_chat([
            {"events": [{"delta": {"content": '[{"name": "a", "arguments": {}}, '
                                               '{"name": "b", "arguments": {"k": 1}}]'}},
                        {"finish_reason": "stop"}]},
        ])
        r = _chat(srv, {"model": "solo-model",
                        "messages": [{"role": "user", "content": "do both"}],
                        "tools": [WEATHER_TOOL]})
        calls = r.json()["choices"][0]["message"]["tool_calls"]
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
        assert json.loads(calls[1]["function"]["arguments"]) == {"k": 1}
    finally:
        srv.stop()


def test_emulation_retries_with_a_followup_when_no_call_was_produced(api_factory, upstream,
                                                                    tmp_path):
    models = single_route_catalogue(tmp_path / "models.json", features=("chat",))
    srv = api_factory(models_path=models)
    try:
        upstream.scenario.set_chat([
            {"events": [{"delta": {"content": "I would rather just explain."}},
                        {"finish_reason": "stop"}]},
            {"events": [{"delta": {"content": '{"tool_call": {"name": "get_weather", '
                                               '"arguments": {"city": "Bergen"}}}'}},
                        {"finish_reason": "stop"}]},
        ])
        r = _chat(srv, {"model": "solo-model",
                        "messages": [{"role": "user", "content": "weather?"}],
                        "tools": [WEATHER_TOOL]})
        calls = r.json()["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Bergen"}
        reqs = _chat_requests(srv)
        assert len(reqs) == 2
        assert "did not include a tool call" in reqs[1]["body"]["messages"][-1]["content"]
    finally:
        srv.stop()


def test_emulation_gives_up_and_returns_the_plain_answer(api_factory, upstream, tmp_path):
    models = single_route_catalogue(tmp_path / "models.json", features=("chat",))
    srv = api_factory(models_path=models)
    try:
        upstream.scenario.set_chat([
            {"events": [{"delta": {"content": "no tool needed"}}, {"finish_reason": "stop"}]},
        ])
        r = _chat(srv, {"model": "solo-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "tools": [WEATHER_TOOL]})
        msg = r.json()["choices"][0]["message"]
        assert msg["content"] == "no tool needed"
        assert "tool_calls" not in msg
    finally:
        srv.stop()


# =========================================================================== #
# message translation
# =========================================================================== #
def test_system_and_tool_messages_are_translated(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    _chat(api, _hello({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "calling", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "12C"},
    ]}))
    payload = _chat_requests(api)[0]["body"]
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "calling"},
        {"role": "assistant", "content": '[called get_weather({"city":"Oslo"})]'},
        {"role": "user", "content": "[get_weather result] 12C"},
    ]


def test_multipart_content_is_forwarded(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    _chat(api, _hello({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
    ]}]}))
    payload = _chat_requests(api)[0]["body"]
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
    ]


def test_long_conversations_are_trimmed(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    messages = [{"role": "system", "content": "sys"}]
    messages += [{"role": "user", "content": f"m{i}"} for i in range(60)]
    _chat(api, _hello({"messages": messages}))
    sent = _chat_requests(api)[0]["body"]["messages"]
    assert len(sent) == 31  # 1 system + the 30 most recent
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[-1] == {"role": "user", "content": "m59"}


def test_oversized_prompts_are_budgeted(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    messages = [{"role": "system", "content": "S" * 20000}]
    messages += [{"role": "user", "content": "u" * 20000} for _ in range(10)]
    _chat(api, _hello({"messages": messages}))
    sent = _chat_requests(api)[0]["body"]["messages"]
    total = sum(len(str(m.get("content") or "")) for m in sent)
    assert total <= 30000 * 4 + 4096 * 10
    assert sent[0]["role"] == "system"


# =========================================================================== #
# failure handling
# =========================================================================== #
def test_rate_limit_is_retried_then_succeeds(api, scenario):
    scenario.set_chat([
        {"status": 429, "body": "slow down"},
        {"events": [{"delta": {"content": "finally"}}, {"finish_reason": "stop"}]},
    ])
    r = _chat(api, _hello({}))
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "finally"
    assert len(_chat_requests(api)) == 2


def test_server_error_is_retried_then_succeeds(api, scenario):
    scenario.set_chat([
        {"status": 500, "body": "boom"},
        {"events": [{"delta": {"content": "recovered"}}, {"finish_reason": "stop"}]},
    ])
    r = _chat(api, _hello({}))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "recovered"
    assert len(_chat_requests(api)) == 2


def test_a_rejected_credential_is_skipped_for_the_next_one(api_factory, upstream, tmp_path,
                                                           scenario):
    srv = api_factory(tokens=["expired-token", "access-token-1"])
    scenario.set_chat([
        {"status": 401, "body": json.dumps({"error": {"message": "website_auth_required",
                                                      "code": "website_auth_required"}})},
        {"events": [{"delta": {"content": "second cred"}}, {"finish_reason": "stop"}]},
    ])
    r = _chat(srv, _hello({}))
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "second cred"
    reqs = _chat_requests(srv)
    assert reqs[0]["headers"]["authorization"] == "Bearer expired-token"
    assert reqs[1]["headers"]["authorization"] == "Bearer access-token-1"

    # a 401 retires the credential for good: it is pruned from the pool
    health = _client(srv).get("/health").json()["credentials"]
    assert health["total"] == 1
    assert health["working"] == 1
    assert [c["email"] for c in health["credentials"]] == ["user1@example.test"]
    srv.stop()


def test_route_scoped_errors_fall_through_to_the_next_route(api, scenario):
    scenario.set_chat([
        {"status": 404, "body": json.dumps({"error": {"message": "model not found",
                                                      "code": "404"}})},
        {"events": [{"delta": {"content": "second route"}}, {"finish_reason": "stop"}]},
    ])
    r = _chat(api, _hello({}))
    assert r.status_code == 200, r.text
    reqs = _chat_requests(api)
    assert len(reqs) == 2
    assert reqs[0]["body"]["provider"] == "voidai"
    assert reqs[1]["body"]["provider"] == "evolvex"


def test_an_in_stream_error_becomes_a_502(api, scenario):
    scenario.set_chat([{"events": [{"error": {"message": "upstream exploded",
                                              "code": "internal"}}]}])
    r = _chat(api, _hello({}))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_error"
    assert "upstream exploded" in r.json()["error"]["message"]


def test_gratisfy_error_text_in_content_is_treated_as_an_error(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "[an error occurred while "
                                                       "processing your request]"}}]}])
    r = _chat(api, _hello({}))
    assert r.status_code == 502
    assert "an error occurred" in r.json()["error"]["message"]


def test_a_repetition_loop_is_rejected(api, scenario):
    events = [{"delta": {"content": "same"}} for _ in range(10)]
    events.append({"finish_reason": "stop"})
    scenario.set_chat([{"events": events}])
    r = _chat(api, _hello({}))
    assert r.status_code == 502
    assert "repetition" in r.json()["error"]["message"].lower()


def test_an_empty_stream_is_a_502(api, scenario):
    """No frames and no [DONE] at all -> the pump only ever sees the sentinel."""
    scenario.set_chat([{"events": [], "suppress_done": True}])
    r = _chat(api, _hello({"stream": True}))
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_empty"


def test_a_stream_with_no_content_gets_a_fallback_message(api, scenario):
    scenario.set_chat([{"events": [{"finish_reason": "stop"}]}])
    with _client(api).stream("POST", "/v1/chat/completions",
                             json=_hello({"stream": True})) as r:
        assert r.status_code == 200
        out = read_sse(r)
    assert out["content"] == "Sorry, the model returned an empty response."
    assert out["frames"][-1]["choices"][0]["finish_reason"] == "stop"


def test_a_non_streaming_empty_response_is_reported_as_null_content(api, scenario):
    scenario.set_chat([{"events": [{"finish_reason": "stop"}]}])
    r = _chat(api, _hello({}))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"] == {"role": "assistant", "content": None}


def test_every_route_failing_is_a_502(api, scenario):
    scenario.set_chat([{"status": 404, "body": "model not found"}])
    r = _chat(api, _hello({}))
    assert r.status_code == 502
    assert "All routes" in r.json()["error"]["message"]


def test_no_credentials_is_a_503(api_factory, upstream, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"credentials": []}))
    srv = api_factory(creds_path=empty)
    try:
        r = _chat(srv, _hello({}))
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "no_credentials"
    finally:
        srv.stop()


def test_expired_credentials_are_refreshed_at_startup(api_factory, upstream, tmp_path):
    creds = tmp_path / "creds.json"
    creds.write_text(json.dumps({"credentials": [
        {"id": "old", "email": "a@b.c", "password": "pw", "access_token": "stale",
         "refresh_token": "rt", "expires_at": time.time() - 10}
    ]}))
    upstream.scenario.set_auth([{"access_token": "renewed", "refresh_token": "rt2",
                                 "expires_in": 7200}])
    srv = api_factory(creds_path=creds, extra_env={"PA_AUTO_REFRESH": "1"})
    try:
        stored = json.loads(creds.read_text())["credentials"]
        assert stored[0]["access_token"] == "renewed"
        assert stored[0]["refresh_token"] == "rt2"
        auth_reqs = [e for e in upstream.log() if e["path"] == "/auth/v1/token"]
        assert auth_reqs and auth_reqs[0]["body"]["refresh_token"] == "rt"
    finally:
        srv.stop()


# =========================================================================== #
# concurrency
# =========================================================================== #
def test_concurrent_streaming_requests(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    results: List[Any] = []
    lock = threading.Lock()

    def one(i: int) -> None:
        with _client(api, timeout=90).stream(
                "POST", "/v1/chat/completions",
                json=_hello({"stream": True, "max_tokens": 8})) as r:
            out = read_sse(r)
        with lock:
            results.append((i, r.status_code, out["content"]))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(results) == 8
    assert all(status == 200 for _, status, _ in results), results
    assert all(content == "ok" for _, _, content in results), results


def test_concurrent_non_streaming_requests(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    results: List[Any] = []
    lock = threading.Lock()

    def one(i: int) -> None:
        r = _chat(api, _hello({}))
        with lock:
            results.append((i, r.status_code, r.json()["choices"][0]["message"]["content"]))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert len(results) == 8
    assert all(s == 200 for _, s, _ in results), results


# =========================================================================== #
# model resolution through the chat endpoint
# =========================================================================== #
@pytest.mark.parametrize("alias", ["kimi-k3", "Kimi K3", "KIMI_K3", "voidai/kimi-k3"])
def test_model_aliases_resolve_through_chat(api, scenario, alias):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    r = _chat(api, _hello({"model": alias}))
    assert r.status_code == 200, (alias, r.text)


def test_chat_without_a_model_uses_the_default(api, scenario):
    scenario.set_chat([{"events": [{"delta": {"content": "ok"}}, {"finish_reason": "stop"}]}])
    r = _client(api).post("/v1/chat/completions",
                          json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert r.json()["model"]  # some default model was chosen
