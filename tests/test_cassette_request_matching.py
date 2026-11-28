"""Cassette replay ignores request fields that change every run (tests/CASSETTE_MAINTENANCE.md).

``prompt_cache_key`` carries the conversation thread id (issue #194), a fresh
UUID on every run, so a body hash that included it could never match a recorded
cassette. Like the output-ceiling fields, it routes or bounds a request without
changing which reply the model gives. That PAL sends it is pinned against the
provider request in ``tests/test_cache_routing_keys.py``.
"""

from __future__ import annotations

import json

import httpx

from tests.http_transport_recorder import ReplayTransport


def _signature(transport: ReplayTransport, body: dict) -> str:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions", content=json.dumps(body))
    return transport._get_request_signature(request)


def _replayer(tmp_path) -> ReplayTransport:
    cassette = tmp_path / "empty.json"
    cassette.write_text(json.dumps({"interactions": []}))
    return ReplayTransport(str(cassette))


def test_prompt_cache_key_does_not_change_the_live_request_signature(tmp_path):
    transport = _replayer(tmp_path)
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}

    first = _signature(transport, {**body, "prompt_cache_key": "thread-a"})
    second = _signature(transport, {**body, "prompt_cache_key": "thread-b"})

    assert first == second == _signature(transport, body)


def test_recorded_and_live_signatures_agree_across_prompt_cache_keys(tmp_path):
    transport = _replayer(tmp_path)
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    recorded = {
        "method": "POST",
        "path": "/v1/chat/completions",
        "content": {**body, "prompt_cache_key": "recorded-thread"},
    }

    live = _signature(transport, {**body, "prompt_cache_key": "this-run-thread"})

    assert transport._get_saved_request_signature(recorded) == live


def test_prompt_text_still_changes_the_signature(tmp_path):
    transport = _replayer(tmp_path)

    hi = _signature(transport, {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]})
    bye = _signature(transport, {"model": "gpt-5", "messages": [{"role": "user", "content": "bye"}]})

    assert hi != bye
