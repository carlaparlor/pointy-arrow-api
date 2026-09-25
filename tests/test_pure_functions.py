"""Pure-function tests for the router's translation layer.

These exercise the real functions with real inputs.
"""
from __future__ import annotations

import json

import pytest

from pa_router import (
    _emulation_messages,
    _flatten_content,
    _is_gratisfy_error_text,
    _is_tool_error,
    _json_position,
    _last_json_object,
    _looks_like_tool_call,
    _looks_route_scoped,
    _parse_plaintext_tool_calls,
    _tool_spec_text,
    to_gratisfy_messages,
)


# --------------------------------------------------------------------------- #
# _flatten_content
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("hello", "hello"),
        (None, ""),
        ([], ""),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        ([{"type": "image_url", "image_url": {"url": "http://x"}}], ""),
        (["plain", {"type": "text", "text": "x"}], "plainx"),
        ([{"type": "text"}], ""),
    ],
)
def test_flatten_content(value, expected):
    assert _flatten_content(value) == expected


# --------------------------------------------------------------------------- #
# to_gratisfy_messages
# --------------------------------------------------------------------------- #
def test_to_gratisfy_messages_passes_plain_text_through():
    out = to_gratisfy_messages([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert out == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_to_gratisfy_messages_collapses_single_text_part():
    out = to_gratisfy_messages([{"role": "user", "content": [{"type": "text", "text": "solo"}]}])
    assert out == [{"role": "user", "content": "solo"}]


def test_to_gratisfy_messages_keeps_multimodal_parts():
    out = to_gratisfy_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
        ]}
    ])
    assert out[0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "https://img/x.png"}},
    ]


def test_to_gratisfy_messages_accepts_a_bare_url_for_images():
    out = to_gratisfy_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": "https://img/y.png"},
        ]}
    ])
    assert out[0]["content"][1] == {"type": "image_url", "image_url": {"url": "https://img/y.png"}}


def test_to_gratisfy_messages_drops_empty_part_lists():
    out = to_gratisfy_messages([{"role": "user", "content": [{"type": "image_url"}]}])
    assert out == [{"role": "user", "content": ""}]


def test_to_gratisfy_messages_flattens_assistant_tool_calls():
    out = to_gratisfy_messages([
        {"role": "assistant", "content": "thinking", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'}},
        ]},
    ])
    assert out[0] == {"role": "assistant", "content": "thinking"}
    assert out[1] == {"role": "assistant", "content": '[called get_weather({"city":"Oslo"})]'}


def test_to_gratisfy_messages_handles_assistant_tool_calls_without_text():
    out = to_gratisfy_messages([
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "a", "arguments": "{}"}},
            {"function": {"name": "b", "arguments": "{}"}},
        ]}
    ])
    assert len(out) == 1
    assert out[0]["content"] == "[called a({}); b({})]"


def test_to_gratisfy_messages_maps_tool_results_to_user():
    out = to_gratisfy_messages([
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "12C"},
        {"role": "tool", "tool_call_id": "c2", "content": "no name"},
    ])
    assert out[0] == {"role": "user", "content": "[get_weather result] 12C"}
    assert out[1] == {"role": "user", "content": "[tool result] no name"}


def test_to_gratisfy_messages_defaults_a_missing_role_to_user():
    assert to_gratisfy_messages([{"content": "x"}]) == [{"role": "user", "content": "x"}]


def test_to_gratisfy_messages_handles_an_empty_conversation():
    assert to_gratisfy_messages([]) == []


# --------------------------------------------------------------------------- #
# _tool_spec_text / _emulation_messages
# --------------------------------------------------------------------------- #
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
]


def test_tool_spec_text_inlines_a_small_tool_list():
    text = _tool_spec_text(TOOLS, None)
    assert "Available tools:" in text
    assert "get_weather" in text
    assert "Get the weather" in text


def test_tool_spec_text_compresses_a_huge_tool_list():
    big = [{"function": {"name": f"tool_{i}", "description": "d" * 500,
                         "parameters": {"required": ["a", "b"]}}} for i in range(50)]
    text = _tool_spec_text(big, None)
    assert len(text) <= 6000 + len("Available tools:\n")
    assert "tool_0(a, b)" in text
    assert "tool_1(a, b)" in text


def test_tool_spec_text_honours_a_named_tool_choice():
    text = _tool_spec_text(TOOLS, {"type": "function", "function": {"name": "get_weather"}})
    assert 'You MUST call the tool named "get_weather"' in text


def test_tool_spec_text_honours_required_tool_choice():
    assert "You MUST call one or more tools." in _tool_spec_text(TOOLS, "required")


def test_emulation_messages_appends_a_system_instruction():
    out = _emulation_messages([{"role": "user", "content": "hi"}], TOOLS, "auto")
    assert len(out) == 2
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "system"
    assert "[TOOL CALLING MODE]" in out[1]["content"]
    assert '"tool_call"' in out[1]["content"]


def test_emulation_messages_does_not_mutate_the_input():
    original = [{"role": "user", "content": "hi"}]
    _emulation_messages(original, TOOLS, "auto")
    assert original == [{"role": "user", "content": "hi"}]


def test_emulation_messages_followup_is_a_different_instruction():
    out = _emulation_messages([{"role": "user", "content": "hi"}], TOOLS, "auto", followup=True)
    assert "did not include a tool call" in out[-1]["content"]
    assert "[TOOL CALLING MODE]" not in out[-1]["content"]


# --------------------------------------------------------------------------- #
# tool-call JSON scraping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("obj,expected", [
    ({"tool_call": {"name": "f", "arguments": {}}}, True),
    ({"name": "f"}, True),
    ({"tool": "f"}, True),
    ({"tool_name": "f"}, True),
    ([{"name": "f"}, {"name": "g"}], True),
    ({"foo": "bar"}, False),
    ([{"foo": 1}], False),
    ("string", False),
    ({"name": 5}, False),
])
def test_looks_like_tool_call(obj, expected):
    assert _looks_like_tool_call(obj) is expected


def test_last_json_object_finds_a_trailing_object():
    text = 'Sure! {"tool_call": {"name": "f", "arguments": {"a": 1}}}'
    obj, idx = _last_json_object(text)
    assert obj == {"tool_call": {"name": "f", "arguments": {"a": 1}}}
    assert text[idx:].startswith("{")


def test_last_json_object_unwraps_a_markdown_fence():
    text = '```json\n{"tool_call": {"name": "f"}}\n```'
    obj, _ = _last_json_object(text)
    assert obj == {"tool_call": {"name": "f"}}


def test_last_json_object_handles_nested_braces_in_strings():
    text = 'blah {"tool_call": {"name": "f", "arguments": {"q": "a{b}c"}}} tail'
    obj, _ = _last_json_object(text)
    assert obj["tool_call"]["arguments"] == {"q": "a{b}c"}


def test_last_json_object_returns_none_for_no_json():
    assert _last_json_object("no json here") == (None, None)
    assert _last_json_object("") == (None, None)


def test_last_json_object_ignores_broken_json():
    obj, idx = _last_json_object('{"tool_call": {"name": ')
    assert obj is None


def test_json_position_matches_the_object_offset():
    text = 'preamble {"tool_call": {"name": "f"}}'
    pos = _json_position(text)
    assert pos is not None
    assert text[pos:].startswith('{"tool_call"')


def test_parse_plaintext_tool_calls_single_object():
    text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}'
    calls = _parse_plaintext_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": '{"city": "Oslo"}'}]


def test_parse_plaintext_tool_calls_bare_shape():
    assert _parse_plaintext_tool_calls('{"name": "f", "arguments": {"x": 1}}') == [
        {"name": "f", "arguments": '{"x": 1}'}]


def test_parse_plaintext_tool_calls_alternate_key_names():
    assert _parse_plaintext_tool_calls('{"tool": "f", "args": {"x": 1}}') == [
        {"name": "f", "arguments": '{"x": 1}'}]
    assert _parse_plaintext_tool_calls('{"tool_name": "f", "parameters": {"x": 1}}') == [
        {"name": "f", "arguments": '{"x": 1}'}]


def test_parse_plaintext_tool_calls_a_list_of_calls():
    text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": 1}}]'
    calls = _parse_plaintext_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == '{"k": 1}'


def test_parse_plaintext_tool_calls_defaults_missing_arguments():
    assert _parse_plaintext_tool_calls('{"name": "f"}') == [{"name": "f", "arguments": "{}"}]


def test_parse_plaintext_tool_calls_ignores_objects_without_a_name():
    assert _parse_plaintext_tool_calls('{"foo": "bar"}') is None
    assert _parse_plaintext_tool_calls("") is None
    assert _parse_plaintext_tool_calls("plain answer, no json") is None


def test_parse_plaintext_tool_calls_skips_unnamed_entries_in_a_list():
    calls = _parse_plaintext_tool_calls('[{"foo": 1}, {"name": "real"}]')
    assert calls == [{"name": "real", "arguments": "{}"}]


def test_parse_plaintext_tool_calls_coerces_non_string_arguments():
    calls = _parse_plaintext_tool_calls('{"name": "f", "arguments": 42}')
    assert calls == [{"name": "f", "arguments": "{}"}]


# --------------------------------------------------------------------------- #
# error classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message", [
    "payment required", "HTTP 402", "insufficient balance", "credit balance exhausted",
    "model not found", "404", "the model does not exist", "unauthorized", "401",
    "403 forbidden", "key limit reached", "billing problem", "entitlement missing",
    "an error occurred", "finish_reason=error", "tool choice is none",
    "tool_use_failed", "function calling unsupported", "not supported",
    "insufficient credits", "input too long",
])
def test_looks_route_scoped_matches_known_markers(message):
    assert _looks_route_scoped(message) is True


@pytest.mark.parametrize("message", [
    "connection reset by peer", "timeout", "temporary glitch", "rate limited: slow down",
    "upstream 500: internal", "",
])
def test_looks_route_scoped_rejects_transient_errors(message):
    assert _looks_route_scoped(message) is False


@pytest.mark.parametrize("message", [
    "tool choice is none", "tool_use_failed", "tool use failed", "function calling not allowed",
    "tool calling unsupported", "not supported", "insufficient credits", "input too long",
])
def test_is_tool_error_matches(message):
    assert _is_tool_error(message) is True


@pytest.mark.parametrize("message", ["", "rate limited", "502 bad gateway", "not found"])
def test_is_tool_error_rejects(message):
    assert _is_tool_error(message) is False


@pytest.mark.parametrize("text,expected", [
    ("[an error occurred", True),
    ("  [AN ERROR OCCURRED while processing]  ", True),
    ("all good here", False),
    ("", False),
])
def test_is_gratisfy_error_text(text, expected):
    assert _is_gratisfy_error_text(text) is expected


def test_tool_spec_text_is_valid_json_for_a_small_list():
    spec = _tool_spec_text(TOOLS, None)
    payload = spec.split("Available tools:\n", 1)[1]
    assert json.loads(payload) == TOOLS
