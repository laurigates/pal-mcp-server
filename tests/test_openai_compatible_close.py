"""Tests for deterministic client shutdown in OpenAICompatibleProvider.

Regression coverage for the leak where the lazily-created AsyncOpenAI client was
never closed, so its httpx connection pool was reclaimed by the garbage collector
after the event loop had already shut down ("RuntimeError: Event loop is closed").
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI

from providers.openai_compatible import OpenAICompatibleProvider


class _StubProvider(OpenAICompatibleProvider):
    """Minimal concrete provider so the abstract base can be instantiated."""

    FRIENDLY_NAME = "Stub"
    MODEL_CAPABILITIES = {"stub-model": {"context_window": 4096}}

    def get_capabilities(self, model_name):
        return MagicMock()

    def get_provider_type(self):
        return MagicMock()

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["stub-model"]


def _provider_with_mock_client():
    """Build a provider whose lazy client is already an awaitable-close mock."""
    provider = _StubProvider("test-key")
    # spec=AsyncOpenAI keeps the mock honest: the real client exposes close(),
    # not aclose(), so a wrong call name fails instead of silently passing.
    client = MagicMock(spec=AsyncOpenAI)
    client.close = AsyncMock()
    provider._client = client
    return provider, client


def test_close_awaits_client_close():
    """close() drives the client's async close to completion."""
    provider, client = _provider_with_mock_client()

    provider.close()

    client.close.assert_awaited_once()


def test_close_clears_client_reference():
    """The client reference is dropped so the next use rebuilds it."""
    provider, _ = _provider_with_mock_client()

    provider.close()

    assert provider._client is None


def test_close_is_idempotent():
    """A second close() is a no-op rather than a double close."""
    provider, client = _provider_with_mock_client()

    provider.close()
    provider.close()

    client.close.assert_awaited_once()


def test_close_without_client_is_noop():
    """Closing a provider that never built a client does not raise."""
    provider = _StubProvider("test-key")
    assert provider._client is None

    provider.close()  # must not raise

    assert provider._client is None


def test_close_survives_a_failing_client():
    """A client that raises on close is logged, not propagated to the caller."""
    provider, client = _provider_with_mock_client()
    client.close = AsyncMock(side_effect=RuntimeError("boom"))

    provider.close()  # must not raise

    assert provider._client is None


@pytest.mark.asyncio
async def test_close_from_running_loop_schedules_cleanup():
    """Called from inside a running loop, close() schedules instead of blocking."""
    provider, client = _provider_with_mock_client()

    provider.close()
    # Fire-and-forget: yield to the loop so the scheduled task can run.
    await asyncio.sleep(0)

    client.close.assert_awaited_once()
    assert provider._client is None
