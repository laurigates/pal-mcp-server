"""Consensus sends the stance after the proposal, not inside the system prompt (issue #194, Stage A).

The stance used to be spliced into the system prompt about 1,100 characters in,
so the same model consulted ``for`` and then ``against`` shared only that much of
its request, and the proposal and context files after it could never come from a
provider cache. Gate G5 in ``tests/test_prompt_prefix_stability.py`` checks the
request bytes end to end; these tests pin the layout.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tools.consensus import ConsensusTool


async def _consult(stance: str, stance_prompt: str | None = None) -> dict:
    tool = ConsensusTool()
    tool.original_proposal = "PROPOSAL: adopt an event log"
    request = Mock(relevant_files=["/src/ledger.py"], images=None, step_number=1, total_steps=2)
    provider = Mock()
    provider.generate_content = Mock(return_value=Mock(content="verdict", usage={}))
    config = {"model": "flash", "stance": stance}
    if stance_prompt:
        config["stance_prompt"] = stance_prompt
    with (
        patch.object(tool, "get_model_provider", return_value=provider),
        patch.object(tool, "_prepare_file_content_for_prompt", return_value=("FILE-BODY", [])),
    ):
        await tool._consult_model(config, request)
    return provider.generate_content.call_args.kwargs


@pytest.mark.asyncio
async def test_system_prompt_is_the_same_for_every_stance():
    for_call = await _consult("for")
    against_call = await _consult("against")
    custom_call = await _consult("for", "Weigh operating cost above all")

    assert for_call["system_prompt"] == against_call["system_prompt"] == custom_call["system_prompt"]
    for marker in ("SUPPORTIVE PERSPECTIVE", "CRITICAL PERSPECTIVE", "BALANCED PERSPECTIVE", "{stance_prompt}"):
        assert marker not in for_call["system_prompt"]


@pytest.mark.asyncio
async def test_stance_follows_the_proposal_and_context_files():
    prompt = (await _consult("against"))["prompt"]

    assert prompt.index("PROPOSAL") < prompt.index("FILE-BODY") < prompt.index("CRITICAL PERSPECTIVE")
    assert prompt.rstrip().endswith("=== END PERSPECTIVE ===")


@pytest.mark.asyncio
async def test_custom_stance_prompt_replaces_the_built_in_stance():
    prompt = (await _consult("for", "Weigh operating cost above all"))["prompt"]

    assert "Weigh operating cost above all" in prompt
    assert "SUPPORTIVE PERSPECTIVE" not in prompt


@pytest.mark.asyncio
async def test_consults_share_everything_before_the_stance():
    for_prompt = (await _consult("for"))["prompt"]
    against_prompt = (await _consult("against"))["prompt"]

    shared = for_prompt.index("=== YOUR PERSPECTIVE ===")
    assert for_prompt[:shared] == against_prompt[:shared]
