"""Per-thread cache routing keys (issue #194, Stage A, docs/prompt-caching.md).

A provider cache lives on the server that handled a request, so consecutive
requests of one thread hit it only when they are routed to the same place.
OpenRouter (``x-session-id``) and OpenAI (``prompt_cache_key``) accept a
routing key for that. PAL sends the thread id, published by
``utils.session_context`` as soon as the thread exists, so turn 1 already carries
the key the later turns will.

The keys go only in the provider-specific ``_build_request`` overrides: a strict
OpenAI-compatible backend may reject a field it does not know.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from providers.custom import CustomProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from utils.session_context import get_session_id, set_session_id

THREAD = "8f3c1f2e-0f3a-4b7a-9f4e-2b6a1c0d5e77"


@pytest.fixture
def thread_session():
    set_session_id(THREAD)
    yield
    set_session_id(None)


def _request(provider, model_name: str) -> dict:
    return provider._build_request(
        prompt="hi", model_name=model_name, system_prompt="sys", temperature=1.0, max_output_tokens=None
    )


@pytest.mark.parametrize("model_name", ["gpt-5", "gpt-6-astra"])  # Chat Completions, Responses
def test_openai_sends_the_thread_as_prompt_cache_key(thread_session, model_name):
    request = _request(OpenAIModelProvider("test-key"), model_name)

    assert request["params"]["prompt_cache_key"] == THREAD


@pytest.mark.parametrize("model_name", ["anthropic/claude-opus-5", "openai/gpt-5.6-sol"])
def test_openrouter_sends_the_thread_as_session_header(thread_session, model_name):
    request = _request(OpenRouterProvider("test-key"), model_name)

    assert request["params"]["extra_headers"]["x-session-id"] == THREAD
    assert "prompt_cache_key" not in request["params"]


def test_the_shared_builder_sends_no_routing_key(thread_session):
    """A strict OpenAI-compatible backend (llama-server, vLLM) may reject unknown fields."""
    provider = CustomProvider(api_key="", base_url="http://localhost:11434/v1")
    with patch.object(provider, "validate_model_name", return_value=True):
        request = _request(provider, "llama3.2")

    assert "prompt_cache_key" not in request["params"]
    assert not request["params"].get("extra_headers")


class TestThreadCreationPublishesTheSessionId:
    """Turn 1 must carry the key: the thread exists before the model is called."""

    def test_simple_tool(self, tmp_path):
        from tools.chat import ChatRequest, ChatTool

        set_session_id(None)
        request = ChatRequest(prompt="q", working_directory_absolute_path=str(tmp_path))
        try:
            thread_id = ChatTool()._start_conversation_thread(request)
            assert thread_id and get_session_id() == thread_id
        finally:
            set_session_id(None)

    def test_workflow_tool(self):
        from tools.thinkdeep import ThinkDeepTool

        set_session_id(None)
        try:
            thread_id = ThinkDeepTool()._create_workflow_thread({"step": "s"}, Mock(step="s"))
            assert get_session_id() == thread_id
        finally:
            set_session_id(None)
