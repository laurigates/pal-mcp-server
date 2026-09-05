"""Concurrent workflow requests through one shared tool instance must not cross-label metadata.

server.py builds TOOLS once and reuses each instance for every request, and the
MCP server runs each request in its own task, so two workflow calls to the same
tool interleave at the expert branch's ``await provider.generate_content``. Any
per-request fact kept on the instance is therefore shared by every in-flight
call — the invariant tests/test_simple_tool_concurrent_metadata.py pins for
simple tools (issue #68). For workflow tools the model and provider a request
called are recorded on that request's own ``arguments`` by
``_call_expert_analysis``, and ``metadata.model_used``/``provider_used`` must
name the model the request's *own* provider was awaited with (issue #96).

Both tests gate request A's ``generate_content`` on an event that request B sets
when its whole ``execute_workflow`` has returned, so B runs to completion while A
is suspended inside its model call.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from providers.shared import ProviderType
from tests.mock_helpers import create_mock_provider
from tools.codereview import CodeReviewTool
from utils.model_context import ModelContext


@pytest.fixture
def reviewed_file(tmp_path):
    """Codereview step 1 requires relevant_files; give it a real absolute path."""
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


def _arguments(model_context: ModelContext, model_name: str, reviewed_file: str, **overrides) -> dict:
    """Arguments as server.py hands them to a workflow tool (model already resolved)."""
    arguments = {
        "step": f"Review {reviewed_file}",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": f"Findings written for {model_name}",
        "relevant_files": [reviewed_file],
        "review_validation_type": "external",
        "model": model_name,
        "_model_context": model_context,
        "_resolved_model_name": model_name,
    }
    arguments.update(overrides)
    return arguments


def _gated_context(model_name: str, provider_type: ProviderType, release: asyncio.Event | None):
    """A ModelContext whose mocked provider answers only after ``release`` is set (at once when None)."""
    provider = create_mock_provider(model_name=model_name)
    provider.get_provider_type.return_value = provider_type
    canned_response = provider.generate_content.return_value

    async def _generate_content(**_kwargs):
        if release is not None:
            await release.wait()
        return canned_response

    provider.generate_content = AsyncMock(side_effect=_generate_content)
    context = ModelContext(model_name)
    context._provider = provider
    return context, provider


def _response(result) -> dict:
    assert len(result) == 1
    return json.loads(result[0].text)


def _audit_fields(response: dict) -> tuple:
    return response["metadata"]["model_used"], response["metadata"]["provider_used"]


@pytest.mark.asyncio
async def test_two_external_reviews_each_name_their_own_model(reviewed_file):
    """A (flash/google) is suspended in its model call while B (pro/openai) runs to completion."""
    tool = CodeReviewTool()
    b_done = asyncio.Event()
    context_a, provider_a = _gated_context("flash", ProviderType.GOOGLE, release=b_done)
    context_b, provider_b = _gated_context("pro", ProviderType.OPENAI, release=None)

    async def run_b():
        try:
            return await tool.execute_workflow(_arguments(context_b, "pro", reviewed_file))
        finally:
            b_done.set()

    result_a, result_b = await asyncio.wait_for(
        asyncio.gather(tool.execute_workflow(_arguments(context_a, "flash", reviewed_file)), run_b()),
        timeout=10,
    )

    # The race must not misroute the requests themselves, only their labels.
    assert provider_a.generate_content.await_args.kwargs["model_name"] == "flash"
    assert provider_b.generate_content.await_args.kwargs["model_name"] == "pro"

    response_a, response_b = _response(result_a), _response(result_b)
    assert response_a["expert_analysis"]["status"] == "analysis_complete"
    assert response_b["expert_analysis"]["status"] == "analysis_complete"

    assert _audit_fields(response_a) == ("flash", "google"), "A must name the model its own provider was awaited with"
    assert _audit_fields(response_b) == ("pro", "openai")


@pytest.mark.asyncio
async def test_external_review_keeps_its_model_when_an_internal_review_interleaves(reviewed_file):
    """B is internal (skip branch, no model) and completes during A's model call; A must still name its model."""
    tool = CodeReviewTool()
    b_done = asyncio.Event()
    context_a, provider_a = _gated_context("flash", ProviderType.GOOGLE, release=b_done)
    context_b, provider_b = _gated_context("pro", ProviderType.OPENAI, release=None)

    async def run_b():
        try:
            return await tool.execute_workflow(
                _arguments(context_b, "pro", reviewed_file, review_validation_type="internal")
            )
        finally:
            b_done.set()

    result_a, result_b = await asyncio.wait_for(
        asyncio.gather(tool.execute_workflow(_arguments(context_a, "flash", reviewed_file)), run_b()),
        timeout=10,
    )

    provider_b.generate_content.assert_not_awaited()
    assert provider_a.generate_content.await_args.kwargs["model_name"] == "flash"

    response_a, response_b = _response(result_a), _response(result_b)
    assert response_a["expert_analysis"]["status"] == "analysis_complete"
    assert response_b["skip_expert_analysis"] is True

    assert _audit_fields(response_a) == ("flash", "google"), "A called flash; B's skip branch must not blank it"
    assert _audit_fields(response_b) == (None, None)
