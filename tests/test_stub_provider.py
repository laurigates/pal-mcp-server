"""Tests for the simulator's stub provider (issue #143).

The stub exists so the tier that gates merges does not wait ~25 minutes on a
CPU-bound model that no scenario asserts on. Its whole value is that it refuses
what a real endpoint refuses: an earlier draft answered POST on any path, and a
simulator run against it passed with a base URL missing its ``/v1`` -- the
misconfiguration #141 had to fix. Each test below pins one refusal, so relaxing
the stub is a test failure rather than a silent loss of coverage.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from simulator_tests.stub_provider import DEFAULT_MODEL, REPLY, serve


@pytest.fixture
def endpoint():
    """A stub on an ephemeral port; yields its base URL."""
    server = serve(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(url: str, payload: dict) -> tuple[int, dict]:
    """POST JSON and return (status, body), treating an HTTP error as a result."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def chat_request(**overrides) -> dict:
    payload = {"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hello"}], "stream": False}
    payload.update(overrides)
    return payload


class TestValidRequests:
    def test_answers_a_well_formed_completion(self, endpoint):
        status, body = post(f"{endpoint}/v1/chat/completions", chat_request())
        assert status == 200
        assert body["choices"][0]["message"] == {"role": "assistant", "content": REPLY}
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["model"] == DEFAULT_MODEL

    def test_reports_usage(self, endpoint):
        """The provider reads usage into its accounting; zeros read as a bug."""
        _, body = post(f"{endpoint}/v1/chat/completions", chat_request())
        usage = body["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_lists_the_model_it_serves(self, endpoint):
        with urllib.request.urlopen(f"{endpoint}/v1/models", timeout=10) as response:
            body = json.loads(response.read())
        assert [entry["id"] for entry in body["data"]] == [DEFAULT_MODEL]


class TestRefusals:
    def test_rejects_a_path_without_the_v1_prefix(self, endpoint):
        """The bug this stub was rewritten to catch: a base URL missing /v1."""
        status, _ = post(f"{endpoint}/chat/completions", chat_request())
        assert status == 404

    def test_rejects_an_unknown_endpoint(self, endpoint):
        status, _ = post(f"{endpoint}/v1/completions", chat_request())
        assert status == 404

    def test_rejects_a_get_on_an_unknown_path(self, endpoint):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{endpoint}/models", timeout=10)
        assert exc.value.code == 404

    def test_rejects_a_body_that_is_not_json(self, endpoint):
        request = urllib.request.Request(
            f"{endpoint}/v1/chat/completions", data=b"not json", headers={"content-type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=10)
        assert exc.value.code == 400

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param(chat_request(model=""), id="no-model"),
            pytest.param(chat_request(model="gpt-4o"), id="unknown-model"),
            pytest.param(chat_request(messages=[]), id="no-messages"),
            pytest.param(chat_request(messages="hello"), id="messages-not-a-list"),
            pytest.param(chat_request(messages=[{"content": "hi"}]), id="message-without-role"),
            pytest.param(chat_request(messages=[{"role": "user"}]), id="message-without-content"),
            pytest.param(chat_request(stream=True), id="streaming-requested"),
        ],
    )
    def test_rejects_malformed_requests(self, endpoint, payload):
        status, body = post(f"{endpoint}/v1/chat/completions", payload)
        assert status == 400
        assert body["error"]["message"]
