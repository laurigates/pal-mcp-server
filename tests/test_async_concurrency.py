"""Async concurrency regression test for provider.generate_content."""

import asyncio
import time  # noqa: F401 — kept for wall-clock measurement

import pytest

from providers.openai_compatible import OpenAICompatibleProvider
from providers.shared import ModelResponse, ProviderType


class _DummyOpenAICompatible(OpenAICompatibleProvider):
    """Minimal concrete subclass to instantiate the abstract base for tests."""

    FRIENDLY_NAME = "DummyOpenAICompatible"

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CUSTOM

    def validate_model_name(self, model_name: str) -> bool:
        return True

    def validate_parameters(self, model_name: str, temperature: float, **kwargs) -> None:  # noqa: ARG002
        return None


def _fake_response(content: str = "ok") -> ModelResponse:
    return ModelResponse(
        content=content,
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        model_name="dummy-model",
        friendly_name="DummyOpenAICompatible",
        provider=ProviderType.CUSTOM,
        metadata={},
    )


@pytest.mark.asyncio
async def test_two_concurrent_generate_content_calls_overlap(monkeypatch):
    """Two 200ms generate_content calls run concurrently must finish well under 350ms.

    This is the Wave A regression guard: if a future change reintroduces a
    blocking SDK call on the event loop, two concurrent calls will serialise
    and the total wall-clock will jump back toward 400ms.
    """
    provider = _DummyOpenAICompatible(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
    )

    async def slow_call_api(_request):
        # Native async sleep — the template method now awaits _call_api directly,
        # so two concurrent calls yield to the event loop and overlap correctly.
        await asyncio.sleep(0.2)
        return object()

    monkeypatch.setattr(provider, "_build_request", lambda *a, **kw: {})
    monkeypatch.setattr(provider, "_call_api", slow_call_api)
    monkeypatch.setattr(
        provider,
        "_parse_response",
        lambda raw, *, model_name, request: _fake_response(),
    )

    start = time.perf_counter()
    results = await asyncio.gather(
        provider.generate_content(prompt="a", model_name="dummy-model"),
        provider.generate_content(prompt="b", model_name="dummy-model"),
    )
    elapsed = time.perf_counter() - start

    assert len(results) == 2
    assert all(r.content == "ok" for r in results)
    assert elapsed < 0.35, (
        f"generate_content calls serialised: took {elapsed:.2f}s for two 200ms operations "
        f"(expected overlap to finish under 0.35s)"
    )


@pytest.mark.asyncio
async def test_sixtyfour_concurrent_calls_break_default_executor_ceiling(monkeypatch):
    """64 concurrent 200ms calls must finish well under the threadpool-bounded floor.

    The previous asyncio.to_thread shim was capped at the default executor's
    32 workers, so 64 calls would serialise into two batches and take ~0.4s+.
    With the native async SDK surface, all 64 calls yield to the event loop
    concurrently and finish in roughly one sleep (~0.2s) plus scheduling
    overhead.
    """
    provider = _DummyOpenAICompatible(
        api_key="test-key",
        base_url="http://localhost:11434/v1",
    )

    async def slow_call_api(_request):
        await asyncio.sleep(0.2)
        return object()

    monkeypatch.setattr(provider, "_build_request", lambda *a, **kw: {})
    monkeypatch.setattr(provider, "_call_api", slow_call_api)
    monkeypatch.setattr(
        provider,
        "_parse_response",
        lambda raw, *, model_name, request: _fake_response(),
    )

    start = time.perf_counter()
    results = await asyncio.gather(
        *(provider.generate_content(prompt=f"p{i}", model_name="dummy-model") for i in range(64))
    )
    elapsed = time.perf_counter() - start

    assert len(results) == 64
    assert all(r.content == "ok" for r in results)
    assert elapsed < 0.35, (
        f"64 concurrent generate_content calls did not overlap: took {elapsed:.2f}s "
        "(would have been ≥0.4s if a 32-worker executor ceiling were in play)"
    )


@pytest.mark.asyncio
async def test_run_with_retries_async_retries_on_retryable_error(monkeypatch):
    """_run_with_retries_async should retry transient errors using asyncio.sleep."""
    provider = _DummyOpenAICompatible(api_key="test-key", base_url="http://localhost:11434/v1")

    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr("providers.base.asyncio.sleep", fake_sleep)

    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient connection reset")
        return _fake_response("final")

    result = await provider._run_with_retries_async(
        operation=flaky,
        max_attempts=4,
        delays=[0.0, 0.0, 0.0],
        log_prefix="test",
    )

    assert result.content == "final"
    assert attempts["n"] == 3
    assert sleep_calls == []  # delays of 0.0 skip asyncio.sleep
