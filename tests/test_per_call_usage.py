"""Workflow expert calls and consensus consults record their own usage per turn (issue #193).

Simple tools already store the provider's usage on the assistant turn. Workflow
expert analysis and consensus consults dropped it, so ``just cache-report`` had
nothing to read for them. Each such turn now carries ``model_metadata["call_usage"]``
naming the model and provider of *that* call and its own usage. Consensus keeps
``accumulated_responses`` across steps, so a cumulative record would count every
earlier consult again; these tests pin that each step records one call.
"""

import json
from unittest.mock import patch

import pytest

from providers.shared import ProviderType
from tests.mock_helpers import create_mock_provider
from tools.codereview import CodeReviewTool
from tools.consensus import ConsensusTool
from utils.conversation_memory import get_thread
from utils.model_context import ModelContext


def _provider(model_name: str, provider_type: ProviderType, usage: dict):
    provider = create_mock_provider(model_name=model_name)
    provider.get_provider_type.return_value = provider_type
    provider.generate_content.return_value.usage = usage
    return provider


def _last_assistant_turn(thread_id: str, tool_name: str):
    thread = get_thread(thread_id)
    assert thread is not None
    turns = [t for t in thread.turns if t.role == "assistant" and t.tool_name == tool_name]
    assert turns, f"no {tool_name} assistant turn on the thread"
    return turns


@pytest.fixture
def reviewed_file(tmp_path):
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


def _codereview_arguments(context: ModelContext, model_name: str, reviewed_file: str, **overrides) -> dict:
    arguments = {
        "step": f"Review {reviewed_file}",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "isAdmin trusts a client-supplied role",
        "relevant_files": [reviewed_file],
        "review_validation_type": "external",
        "model": model_name,
        "_model_context": context,
        "_resolved_model_name": model_name,
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.asyncio
async def test_workflow_expert_call_records_its_usage_on_the_turn(reviewed_file):
    usage = {"input_tokens": 4000, "output_tokens": 300, "total_tokens": 4300, "cached_input_tokens": 3072}
    context = ModelContext("flash")
    context._provider = _provider("flash", ProviderType.GOOGLE, usage)

    result = await CodeReviewTool().execute_workflow(_codereview_arguments(context, "flash", reviewed_file))
    response = json.loads(result[0].text)

    (turn,) = _last_assistant_turn(response["continuation_id"], "codereview")
    assert turn.model_metadata["call_usage"] == {"model_name": "flash", "provider": "google", "usage": usage}
    # The workflow state the next step restores from is still there.
    assert "work_history" in turn.model_metadata


@pytest.mark.asyncio
async def test_workflow_step_without_a_model_call_records_no_usage(reviewed_file):
    context = ModelContext("flash")
    context._provider = _provider("flash", ProviderType.GOOGLE, {"input_tokens": 1})

    result = await CodeReviewTool().execute_workflow(
        _codereview_arguments(context, "flash", reviewed_file, review_validation_type="internal")
    )
    response = json.loads(result[0].text)

    (turn,) = _last_assistant_turn(response["continuation_id"], "codereview")
    assert "call_usage" not in turn.model_metadata


@pytest.mark.asyncio
async def test_consensus_records_each_consult_once():
    providers = {
        "flash": _provider("flash", ProviderType.GOOGLE, {"input_tokens": 1000, "cached_input_tokens": 0}),
        "o3": _provider("o3", ProviderType.OPENAI, {"input_tokens": 2000, "cached_input_tokens": 1024}),
    }
    step = {
        "step": "Should we adopt the strangler pattern for the billing service?",
        "total_steps": 2,
        "findings": "Billing is a monolith with three teams contending for it",
        "models": [{"model": "flash", "stance": "for"}, {"model": "o3", "stance": "against"}],
    }

    with patch.object(ConsensusTool, "get_model_provider", side_effect=lambda name: providers[name]):
        first = await ConsensusTool().execute_workflow({**step, "step_number": 1, "next_step_required": True})
        thread_id = json.loads(first[0].text)["continuation_offer"]["continuation_id"]
        await ConsensusTool().execute_workflow(
            {
                **step,
                "step": "flash argued for it",
                "step_number": 2,
                "next_step_required": False,
                "continuation_id": thread_id,
            }
        )

    turn_1, turn_2 = _last_assistant_turn(thread_id, "consensus")
    assert turn_1.model_metadata["call_usage"] == {
        "model_name": "flash",
        "provider": "google",
        "usage": {"input_tokens": 1000, "cached_input_tokens": 0},
    }
    assert turn_2.model_metadata["call_usage"] == {
        "model_name": "o3",
        "provider": "openai",
        "usage": {"input_tokens": 2000, "cached_input_tokens": 1024},
    }
    # Usage stays out of the accumulated responses that the client sees and the
    # thread carries forward, so nothing reading them counts a consult twice.
    for accumulated in turn_2.model_metadata["tool_state"]["accumulated_responses"]:
        assert "usage" not in accumulated
