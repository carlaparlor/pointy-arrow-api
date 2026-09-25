"""Regression tests for parallel tool-call extraction (was dropping all but one)."""
from __future__ import annotations

import pytest

from pa_router import _json_position, _last_json_object, _parse_plaintext_tool_calls


def test_a_list_of_calls_is_kept_whole():
    text = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": 1}}]'
    calls = _parse_plaintext_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == '{"k": 1}'


def test_a_wrapped_list_of_calls_is_kept_whole():
    text = '{"tool_calls": [{"name": "a"}, {"name": "b"}]}'
    calls = _parse_plaintext_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_the_outermost_payload_wins_over_a_nested_one():
    text = '{"tool_call": {"name": "outer", "arguments": {"name": "inner"}}}'
    calls = _parse_plaintext_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "outer"


def test_prose_before_a_list_of_calls_still_parses():
    text = 'I will call both tools now. [{"name": "a"}, {"name": "b"}]'
    calls = _parse_plaintext_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_preamble_offset_points_at_the_outermost_object():
    text = 'Sure! [{"name": "a"}, {"name": "b"}]'
    pos = _json_position(text)
    assert pos is not None
    assert text[pos:].startswith("[")


def test_single_object_still_parses():
    assert _parse_plaintext_tool_calls('{"name": "solo"}') == [
        {"name": "solo", "arguments": "{}"}]


def test_three_calls_survive():
    text = '[{"name": "a"}, {"name": "b"}, {"name": "c"}]'
    assert [c["name"] for c in _parse_plaintext_tool_calls(text)] == ["a", "b", "c"]


def test_empty_list_is_not_a_tool_call():
    assert _parse_plaintext_tool_calls("[]") is None
    assert _last_json_object("[]")[0] == []


def test_fenced_list_of_calls():
    text = '```json\n[{"name": "a"}, {"name": "b"}]\n```'
    assert [c["name"] for c in _parse_plaintext_tool_calls(text)] == ["a", "b"]
