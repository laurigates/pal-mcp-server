#!/usr/bin/env python3
"""A strict OpenAI-compatible endpoint that answers without a model.

The simulator scenarios marked ``provider_agnostic`` assert on conversation
threading, file handling and the JSON-RPC boundary. None of them asserts on
what a reply *says* -- they check that it is non-empty, that the server minted a
``continuation_id``, and that the server's own debug log shows the file
embedding and deduplication lines. Measured against issue #143, that means the
model is ~97% of the suite's runtime and 0% of its assertions:

    strict stub        21s     smollm2:135m    2m42s
    qwen2.5:0.5b     4m10s     llama3.2:1b    12m08s

all four with the same ``4/4 tests passed``.

So this stands in for the model on the tier that gates merges, and a real Ollama
keeps the end-to-end signal on a slower tier where nobody is waiting.

**Strictness is the whole design.** A stub that accepts anything tests nothing:
an earlier draft of this file answered ``POST`` on any path, and a run against it
passed with ``CUSTOM_API_URL`` missing its ``/v1`` suffix -- the exact
misconfiguration #141 had to fix, where the documented URL 404s against real
Ollama. Every check below exists because dropping it would make some real defect
invisible. Add to them rather than relax them; anything this file accepts is
something CI can no longer catch.

Usage::

    python -m simulator_tests.stub_provider --port 11500 &
    export CUSTOM_API_URL=http://localhost:11500/v1
    export CUSTOM_MODELS_CONFIG_PATH=simulator_tests/conf/stub_models.json
    export SIMULATOR_MODEL=ci-local DEFAULT_MODEL=ci-local
    uv run python communication_simulator_test.py --ci
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Model id this endpoint serves. Must match ``model_name`` in the registry
#: pointed at by CUSTOM_MODELS_CONFIG_PATH, since the Custom provider sends the
#: resolved name and this endpoint refuses anything else.
DEFAULT_MODEL = "stub-model"

#: What the assistant "says". Deliberately useless: a scenario that starts
#: depending on reply content should fail here rather than pass by luck.
REPLY = "Stub response. This endpoint serves the simulator; it runs no model."


class StubHandler(BaseHTTPRequestHandler):
    """Serves ``/v1/models`` and ``/v1/chat/completions``, and nothing else."""

    protocol_version = "HTTP/1.1"

    # Set by serve() so every handler instance shares one config and log.
    model: str = DEFAULT_MODEL
    requests: list[dict] = []
    lock = threading.Lock()

    def log_message(self, *_args):
        """Silence the per-request stderr line; the caller has its own logs."""

    # -- plumbing ----------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, status: int, message: str) -> None:
        """Refuse a request the real API would refuse, in its error shape."""
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    # -- endpoints ---------------------------------------------------------

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.rstrip("/") != "/v1/models":
            self._reject(404, f"no such endpoint: {self.path}")
            return
        self._send_json(
            200,
            {
                "object": "list",
                "data": [{"id": self.model, "object": "model", "created": 0, "owned_by": "simulator-stub"}],
            },
        )

    def do_POST(self):  # noqa: N802
        # The endpoint's path is part of the contract: Ollama, vLLM and OpenAI
        # all serve chat completions under /v1, and a base URL without it is a
        # configuration bug this endpoint has to surface rather than absorb.
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._reject(404, f"no such endpoint: {self.path}")
            return

        try:
            length = int(self.headers.get("content-length", 0))
        except ValueError:
            self._reject(400, "content-length is not an integer")
            return
        if length <= 0:
            self._reject(400, "empty request body")
            return

        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._reject(400, f"body is not JSON: {exc}")
            return
        if not isinstance(request, dict):
            self._reject(400, "body is not a JSON object")
            return

        problem = self._validate(request)
        if problem:
            self._reject(400, problem)
            return

        with self.lock:
            self.requests.append(request)

        # Token counts are fabricated but internally consistent: the provider
        # reads usage into its own accounting, and a zero there has been
        # mistaken for a provider bug before.
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in request["messages"]) // 4
        completion_tokens = len(REPLY) // 4
        self._send_json(
            200,
            {
                "id": "chatcmpl-simulator-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": REPLY},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _validate(self, request: dict) -> str | None:
        """Return why the real API would reject this request, or None.

        Only what a real endpoint genuinely enforces: an unknown model, a
        missing or malformed ``messages`` array, or a request asking for a
        streamed response this endpoint does not implement.
        """
        model = request.get("model")
        if not model:
            return "missing 'model'"
        if model != self.model:
            return f"unknown model {model!r}; this endpoint serves {self.model!r}"

        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return "'messages' must be a non-empty array"
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                return f"messages[{index}] is not an object"
            if not message.get("role"):
                return f"messages[{index}] has no role"
            if "content" not in message:
                return f"messages[{index}] has no content"

        if request.get("stream"):
            return "streaming is not implemented by this endpoint"
        return None


def serve(port: int = 11500, model: str = DEFAULT_MODEL) -> ThreadingHTTPServer:
    """Start the endpoint on ``port`` and return the server, already listening."""
    StubHandler.model = model
    StubHandler.requests = []
    return ThreadingHTTPServer(("127.0.0.1", port), StubHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=11500, help="port to listen on (default: 11500)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model id to serve (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    server = serve(port=args.port, model=args.model)
    print(f"stub provider serving {args.model} on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
