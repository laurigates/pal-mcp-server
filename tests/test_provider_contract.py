"""Cross-provider contract tests for the generate_content template method.

The template method on :class:`providers.base.ModelProvider` orchestrates the
three abstract hooks ``_build_request`` / ``_call_api`` / ``_parse_response``
and handles retries with progressive delays. These tests assert that every
concrete provider honours that contract:

* ``_build_request`` is called exactly once before the SDK invocation
* retryable failures from ``_call_api`` trigger up to ``MAX_RETRIES`` attempts
* a successful response is returned as a populated :class:`ModelResponse`
"""

import os
import sys
import types

import pytest

from providers.shared import ModelResponse, ProviderType

if "openai" not in sys.modules:  # pragma: no cover - shim for optional Azure dep
    _stub = types.ModuleType("openai")
    _stub.AsyncAzureOpenAI = object
    sys.modules["openai"] = _stub


def _make_response(model_name: str, provider: ProviderType) -> ModelResponse:
    return ModelResponse(
        content="hello",
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        model_name=model_name,
        friendly_name="contract-test",
        provider=provider,
        metadata={"finish_reason": "stop"},
    )


def _gemini_provider():
    from providers.gemini import GeminiModelProvider

    return GeminiModelProvider("test-key"), "gemini-2.5-flash", ProviderType.GOOGLE


def _openai_provider():
    from providers.openai import OpenAIModelProvider

    return OpenAIModelProvider("test-key"), "gpt-4.1", ProviderType.OPENAI


def _dial_provider():
    os.environ.pop("DIAL_ALLOWED_MODELS", None)
    from providers.dial import DIALModelProvider

    return DIALModelProvider("test-key"), "o3", ProviderType.DIAL


def _azure_provider():
    from providers.azure_openai import AzureOpenAIProvider

    return (
        AzureOpenAIProvider(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com/",
            deployments={"gpt-4o": "prod-gpt4o"},
        ),
        "gpt-4o",
        ProviderType.AZURE,
    )


PROVIDER_FACTORIES = [
    pytest.param(_gemini_provider, id="gemini"),
    pytest.param(_openai_provider, id="openai"),
    pytest.param(_dial_provider, id="dial"),
    pytest.param(_azure_provider, id="azure"),
]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip the retry sleep so contract tests run fast."""

    async def _noop(_delay):
        return None

    monkeypatch.setattr("providers.base.asyncio.sleep", _noop)


@pytest.mark.parametrize("factory", PROVIDER_FACTORIES)
async def test_build_request_called_once_before_api(monkeypatch, factory):
    """The template should call ``_build_request`` exactly once per generate_content."""

    provider, model_name, provider_type = factory()

    build_calls = {"count": 0}
    api_calls = {"count": 0}

    original_build = provider._build_request

    def tracking_build(*args, **kwargs):
        build_calls["count"] += 1
        api_calls_at_build = api_calls["count"]
        assert api_calls_at_build == 0, "_build_request must run before _call_api"
        # Return a sentinel; downstream hooks are stubbed so contents don't matter.
        return {"__sentinel__": True}

    async def stub_call_api(_request):
        api_calls["count"] += 1
        return _make_response(model_name, provider_type)

    def stub_parse_response(raw, *, model_name, request):
        assert raw is not None
        return raw

    monkeypatch.setattr(provider, "_build_request", tracking_build)
    monkeypatch.setattr(provider, "_call_api", stub_call_api)
    monkeypatch.setattr(provider, "_parse_response", stub_parse_response)

    response = await provider.generate_content(prompt="hi", model_name=model_name)

    assert build_calls["count"] == 1
    assert api_calls["count"] == 1
    assert isinstance(response, ModelResponse)
    assert response.provider is provider_type

    # Reference original_build to keep the binding active (silences linters).
    assert callable(original_build)


@pytest.mark.parametrize("factory", PROVIDER_FACTORIES)
async def test_retry_uses_max_retries_on_retryable_failures(monkeypatch, factory):
    """Retryable errors must be retried up to MAX_RETRIES attempts."""

    provider, model_name, provider_type = factory()
    monkeypatch.setattr(provider, "_is_error_retryable", lambda _exc: True)

    attempts = {"count": 0}

    async def failing_call_api(_request):
        attempts["count"] += 1
        raise RuntimeError("transient")

    monkeypatch.setattr(provider, "_build_request", lambda *a, **kw: {})
    monkeypatch.setattr(provider, "_call_api", failing_call_api)
    monkeypatch.setattr(
        provider,
        "_parse_response",
        lambda raw, *, model_name, request: _make_response(model_name, provider_type),
    )

    # Per-provider wrap convention varies (RuntimeError / ValueError) — assert either.
    with pytest.raises((RuntimeError, ValueError)):
        await provider.generate_content(prompt="hi", model_name=model_name)

    assert attempts["count"] == provider.MAX_RETRIES


@pytest.mark.parametrize("factory", PROVIDER_FACTORIES)
async def test_non_retryable_error_bails_immediately(monkeypatch, factory):
    """Non-retryable errors must surface on the first attempt."""

    provider, model_name, _provider_type = factory()
    monkeypatch.setattr(provider, "_is_error_retryable", lambda _exc: False)

    attempts = {"count": 0}

    async def failing_call_api(_request):
        attempts["count"] += 1
        raise RuntimeError("permanent failure")

    monkeypatch.setattr(provider, "_build_request", lambda *a, **kw: {})
    monkeypatch.setattr(provider, "_call_api", failing_call_api)
    monkeypatch.setattr(provider, "_parse_response", lambda raw, **_: raw)

    # Per-provider wrap convention varies (RuntimeError / ValueError) — assert either.
    with pytest.raises((RuntimeError, ValueError)):
        await provider.generate_content(prompt="hi", model_name=model_name)

    assert attempts["count"] == 1


@pytest.mark.parametrize("factory", PROVIDER_FACTORIES)
async def test_successful_response_returns_model_response(monkeypatch, factory):
    """Happy path returns a populated ModelResponse from _parse_response."""

    provider, model_name, provider_type = factory()

    async def stub_call_api_happy(_request):
        return object()

    monkeypatch.setattr(provider, "_build_request", lambda *a, **kw: {})
    monkeypatch.setattr(provider, "_call_api", stub_call_api_happy)
    monkeypatch.setattr(
        provider,
        "_parse_response",
        lambda raw, *, model_name, request: _make_response(model_name, provider_type),
    )

    response = await provider.generate_content(prompt="hi", model_name=model_name)

    assert isinstance(response, ModelResponse)
    assert response.content == "hello"
    assert response.model_name == model_name
    assert response.provider is provider_type
