"""
thinkdeep's problem_context and focus_areas reach the expert model (issue #201).

The framing used to live in customize_expert_analysis_prompt, which nothing
called. These tests drive the real expert call with a stub provider and read
the prompt it was handed.
"""

import json
from unittest.mock import Mock

import pytest

from tests.mock_helpers import create_mock_provider
from tools.thinkdeep import ThinkDeepTool


def _stub_model(tool: ThinkDeepTool, monkeypatch) -> Mock:
    provider = create_mock_provider()
    context = Mock()
    context.provider = provider
    context.capabilities = provider.get_capabilities.return_value

    def resolve(arguments, request):
        tool._model_context = context
        tool._current_model_name = "gemini-2.5-flash"
        return "gemini-2.5-flash", context

    monkeypatch.setattr(tool, "_resolve_model_context", resolve)
    return provider


def _step(number: int, total: int, **overrides) -> dict:
    args = {
        "step": f"Step {number}: weigh the two cache designs",
        "step_number": number,
        "total_steps": total,
        "next_step_required": number < total,
        "findings": f"FINDING-{number}",
        "confidence": "high",
    }
    args.update(overrides)
    return args


def _expert_prompt(provider: Mock) -> str:
    provider.generate_content.assert_awaited_once()
    return provider.generate_content.await_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_expert_request_carries_problem_context_and_focus_areas(monkeypatch):
    tool = ThinkDeepTool()
    provider = _stub_model(tool, monkeypatch)

    await tool.execute(
        _step(1, 1, problem_context="PROBLEM-CTX: read-heavy service", focus_areas=["FOCUS-PERF", "FOCUS-SEC"])
    )

    prompt = _expert_prompt(provider)
    assert "PROBLEM-CTX: read-heavy service" in prompt
    assert "FOCUS-PERF, FOCUS-SEC" in prompt


@pytest.mark.asyncio
async def test_stable_scope_precedes_volatile_confidence_and_step_count(monkeypatch):
    """Confidence and step counts change between calls; they come after the
    stable scope so the prompt prefix stays cacheable (docs/prompt-caching.md)."""
    tool = ThinkDeepTool()
    provider = _stub_model(tool, monkeypatch)

    await tool.execute(_step(1, 1, problem_context="PROBLEM-CTX", focus_areas=["FOCUS-PERF"]))

    prompt = _expert_prompt(provider)
    assert prompt.index("PROBLEM-CTX") < prompt.index("Final confidence")
    assert prompt.index("FOCUS-PERF") < prompt.index("Steps completed")


@pytest.mark.asyncio
async def test_scope_given_on_step_one_survives_to_the_expert_step(monkeypatch):
    """Agents pass problem_context on step 1 and omit it later; the expert call
    happens on the last step, so the scope must carry forward on the thread."""
    first = json.loads((await ThinkDeepTool().execute(_step(1, 2, problem_context="PROBLEM-CTX")))[0].text)

    tool = ThinkDeepTool()
    provider = _stub_model(tool, monkeypatch)
    await tool.execute(_step(2, 2, continuation_id=first["continuation_id"], focus_areas=["FOCUS-LATE"]))

    prompt = _expert_prompt(provider)
    assert "PROBLEM-CTX" in prompt
    assert "FOCUS-LATE" in prompt
