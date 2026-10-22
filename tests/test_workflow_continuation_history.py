"""Workflow continuations skip the history build nothing reads (issue #202).

``reconstruct_thread_context`` built the history-enhanced prompt on every
continuation, re-reading every file the thread referenced. Workflow tools take
their input from ``step`` and never read ``prompt``, so for them the build is
skipped. They still get the resolved ``_model_context``.
"""

from unittest.mock import patch

import pytest

from server import reconstruct_thread_context
from tests.mock_helpers import create_mock_provider
from utils.conversation_memory import add_turn, create_thread
from utils.model_context import ModelContext


def _thread(tool_name):
    thread_id = create_thread(tool_name, {"step": "first"})
    add_turn(thread_id, "user", "first", tool_name=tool_name)
    add_turn(thread_id, "assistant", "reply", tool_name=tool_name, model_name="flash", model_provider="google")
    return thread_id


def _arguments(thread_id, **fields):
    context = ModelContext("flash")
    context._provider = create_mock_provider(model_name="flash")
    return {"continuation_id": thread_id, "model": "flash", "_model_context": context, **fields}


def _workflow_fields():
    return {"step": "Next step", "step_number": 2, "total_steps": 2, "next_step_required": False, "findings": "x"}


class TestWorkflowContinuationSkipsHistory:
    @pytest.mark.asyncio
    async def test_workflow_continuation_does_not_build_history(self):
        arguments = _arguments(_thread("codereview"), **_workflow_fields())

        with patch("utils.context_builder.build_conversation_history") as build:
            enhanced = await reconstruct_thread_context(arguments, tool_name="codereview")

        build.assert_not_called()
        assert enhanced["_model_context"] is arguments["_model_context"]
        assert "prompt" not in enhanced
        assert "_remaining_tokens" not in enhanced

    @pytest.mark.asyncio
    async def test_workflow_continuing_a_chat_thread_does_not_build_history(self):
        arguments = _arguments(_thread("chat"), **_workflow_fields())

        with patch("utils.context_builder.build_conversation_history") as build:
            await reconstruct_thread_context(arguments, tool_name="codereview")

        build.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_continuing_a_workflow_thread_still_builds_history(self):
        arguments = _arguments(_thread("codereview"), prompt="Follow-up")

        with patch("utils.context_builder.build_conversation_history", return_value=("HISTORY", 10)) as build:
            enhanced = await reconstruct_thread_context(arguments, tool_name="chat")

        build.assert_called_once()
        assert enhanced["prompt"].startswith("HISTORY")
