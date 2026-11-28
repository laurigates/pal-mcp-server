"""RecordingTransport must record through an async httpx client (#260).

The OpenAI provider builds an ``httpx.AsyncClient``, which calls
``handle_async_request``. RecordingTransport implemented only the sync
``handle_request``, so every OpenAI recording attempt failed with
AttributeError and retried into "Connection error", while replay (a
MockTransport, sync and async) kept passing.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from tests.http_transport_recorder import RecordingTransport, ReplayTransport

BODY = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
REPLY = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}


def _upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=REPLY)


async def _post(transport: httpx.AsyncBaseTransport) -> httpx.Response:
    async with httpx.AsyncClient(transport=transport) as client:
        return await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=BODY,
            headers={"Authorization": "Bearer sk-live-secret-value"},
        )


def test_async_client_records_through_the_transport(tmp_path):
    cassette = tmp_path / "async.json"
    transport = RecordingTransport(str(cassette), async_upstream=httpx.MockTransport(_upstream))

    response = asyncio.run(_post(transport))

    assert response.status_code == 200
    assert response.json() == REPLY
    saved = json.loads(cassette.read_text())["interactions"]
    assert len(saved) == 1
    assert saved[0]["request"]["content"] == BODY
    assert "sk-live-secret-value" not in cassette.read_text()


def test_recording_drops_routing_keys_from_the_request_body(tmp_path):
    # prompt_cache_key is the thread id, a fresh UUID per run that replay
    # ignores; written to a cassette it only trips gitleaks' generic-api-key rule.
    cassette = tmp_path / "async.json"
    transport = RecordingTransport(str(cassette), async_upstream=httpx.MockTransport(_upstream))

    async def _post_with_routing_key() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.openai.com/v1/chat/completions",
                json={**BODY, "prompt_cache_key": "thread-under-test"},
            )

    asyncio.run(_post_with_routing_key())

    saved = json.loads(cassette.read_text())["interactions"]
    assert saved[0]["request"]["content"] == BODY


def test_an_async_recording_replays(tmp_path):
    cassette = tmp_path / "async.json"
    asyncio.run(_post(RecordingTransport(str(cassette), async_upstream=httpx.MockTransport(_upstream))))

    replayed = asyncio.run(_post(ReplayTransport(str(cassette))))

    assert replayed.json() == REPLY
