"""Concurrent requests through one shared tool instance must not cross-label metadata.

server.py builds TOOLS once and reuses each instance for every request, so any
per-request state kept on the instance is shared by all in-flight calls. A slow
OpenAI-routed chat used to report model_used from whatever a later, faster call
had written, producing an inconsistent (requested, provider, model_used) triple.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from providers.shared import ProviderType
from tools.chat import ChatTool


class _FakeProvider:
    """Provider that answers after a caller-controlled delay."""

    def __init__(self, provider_type: ProviderType, delay: float):
        self._provider_type = provider_type
        self._delay = delay
        self.called_with_models: list[str] = []

    def get_provider_type(self) -> ProviderType:
        return self._provider_type

    async def generate_content(self, *, model_name, **kwargs):  # noqa: ARG002
        self.called_with_models.append(model_name)
        await asyncio.sleep(self._delay)
        return SimpleNamespace(
            content=f"answer from {model_name}",
            usage=None,
            metadata={},
        )


def _model_context(model_name: str, provider: _FakeProvider):
    """Minimal stand-in for utils.model_context.ModelContext."""
    capabilities = SimpleNamespace(
        supports_extended_thinking=False,
        allow_code_generation=False,
        supports_images=False,
        temperature_constraint=None,
    )
    return SimpleNamespace(
        model_name=model_name,
        provider=provider,
        capabilities=capabilities,
        calculate_token_allocation=lambda: SimpleNamespace(file_tokens=50_000, history_tokens=50_000),
        estimate_tokens=lambda text: len(text) // 4,
    )


async def _run(tool: ChatTool, model_name: str, provider: _FakeProvider, tmp_path):
    return await tool.execute(
        {
            "prompt": f"question for {model_name}",
            "model": model_name,
            "working_directory_absolute_path": str(tmp_path),
            "_model_context": _model_context(model_name, provider),
        }
    )


def _metadata(result):
    import json

    return json.loads(result[0].text).get("metadata") or {}


@pytest.mark.asyncio
async def test_slow_call_reports_its_own_model(tmp_path):
    """The slow call must report the model it actually asked for."""
    tool = ChatTool()
    slow = _FakeProvider(ProviderType.OPENAI, delay=0.15)
    fast = _FakeProvider(ProviderType.GOOGLE, delay=0.0)

    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        slow_result, _ = await asyncio.gather(
            _run(tool, "gpt-5.3-codex", slow, tmp_path),
            _run(tool, "gemini-3.5-flash", fast, tmp_path),
        )

    assert _metadata(slow_result)["model_used"] == "gpt-5.3-codex"


@pytest.mark.asyncio
async def test_slow_call_reports_a_consistent_model_and_provider(tmp_path):
    """model_used and provider_used must describe the same call."""
    tool = ChatTool()
    slow = _FakeProvider(ProviderType.OPENAI, delay=0.15)
    fast = _FakeProvider(ProviderType.GOOGLE, delay=0.0)

    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        slow_result, _ = await asyncio.gather(
            _run(tool, "gpt-5.3-codex", slow, tmp_path),
            _run(tool, "gemini-3.5-flash", fast, tmp_path),
        )

    metadata = _metadata(slow_result)
    assert (metadata["model_used"], metadata["provider_used"]) == ("gpt-5.3-codex", "openai")


@pytest.mark.asyncio
async def test_fast_call_reports_its_own_model(tmp_path):
    """The call that finished first is labelled correctly too."""
    tool = ChatTool()
    slow = _FakeProvider(ProviderType.OPENAI, delay=0.15)
    fast = _FakeProvider(ProviderType.GOOGLE, delay=0.0)

    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        _, fast_result = await asyncio.gather(
            _run(tool, "gpt-5.3-codex", slow, tmp_path),
            _run(tool, "gemini-3.5-flash", fast, tmp_path),
        )

    metadata = _metadata(fast_result)
    assert (metadata["model_used"], metadata["provider_used"]) == ("gemini-3.5-flash", "google")


@pytest.mark.asyncio
async def test_each_provider_is_called_with_its_own_model(tmp_path):
    """The race must not misroute the request itself, only its label."""
    tool = ChatTool()
    slow = _FakeProvider(ProviderType.OPENAI, delay=0.15)
    fast = _FakeProvider(ProviderType.GOOGLE, delay=0.0)

    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        await asyncio.gather(
            _run(tool, "gpt-5.3-codex", slow, tmp_path),
            _run(tool, "gemini-3.5-flash", fast, tmp_path),
        )

    assert slow.called_with_models == ["gpt-5.3-codex"]
    assert fast.called_with_models == ["gemini-3.5-flash"]
