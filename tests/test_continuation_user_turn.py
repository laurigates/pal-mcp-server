"""A continuation routed through server.py records one user turn (issue #187).

``reconstruct_thread_context`` records the user turn and hands the tool a
prompt with the history already embedded. The tool used to miss the marker
that history opens with, so it recorded the user turn a second time (holding
the whole enhanced prompt) and built the history again around it.

The fix must keep the tool's own prompt preparation on this path: a file or
directory attached on the continuation still reaches the model, and an
oversize prompt is still refused (#194).
"""

import json
from unittest.mock import patch

import pytest

from server import reconstruct_thread_context
from tests.mock_helpers import create_mock_provider
from tools.chat import ChatTool
from tools.shared.exceptions import ToolExecutionError
from utils.conversation_memory import (
    CONVERSATION_HISTORY_MARKER,
    NEW_USER_INPUT_MARKER,
    add_turn,
    create_thread,
    get_thread,
)
from utils.model_context import ModelContext

FIRST_PROMPT = "What is in the config loader?"
FIRST_REPLY = "It reads YAML."
FOLLOW_UP = "And how does it handle missing keys?"


def _model_context(provider):
    context = ModelContext("flash")
    context._provider = provider
    return context


def _chat_thread(tmp_path):
    thread_id = create_thread("chat", {"prompt": FIRST_PROMPT, "working_directory_absolute_path": str(tmp_path)})
    add_turn(thread_id, "user", FIRST_PROMPT, tool_name="chat")
    add_turn(thread_id, "assistant", FIRST_REPLY, tool_name="chat", model_name="flash", model_provider="google")
    return thread_id


async def _continue(tmp_path, thread_id, provider, prompt=FOLLOW_UP, files=None):
    arguments = {
        "prompt": prompt,
        "continuation_id": thread_id,
        "model": "flash",
        "working_directory_absolute_path": str(tmp_path),
        "_model_context": _model_context(provider),
    }
    if files is not None:
        arguments["absolute_file_paths"] = files
    enhanced = await reconstruct_thread_context(arguments)
    tool = ChatTool()
    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        result = await tool.execute(enhanced)
    return json.loads(result[0].text)


def _sent_prompt(provider):
    return provider.generate_content.call_args.kwargs["prompt"]


class TestServerContinuationRecordsOneUserTurn:
    @pytest.mark.asyncio
    async def test_thread_gains_exactly_one_user_turn_with_the_original_prompt(self, tmp_path):
        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        await _continue(tmp_path, thread_id, provider)

        new_user_turns = [turn for turn in get_thread(thread_id).turns[2:] if turn.role == "user"]
        assert [turn.content for turn in new_user_turns] == [FOLLOW_UP]

    @pytest.mark.asyncio
    async def test_history_is_embedded_once(self, tmp_path):
        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        await _continue(tmp_path, thread_id, provider)

        prompt = _sent_prompt(provider)
        assert prompt.count(CONVERSATION_HISTORY_MARKER) == 1
        assert FOLLOW_UP in prompt
        assert FIRST_REPLY in prompt


class TestServerContinuationKeepsPromptPreparation:
    @pytest.mark.asyncio
    async def test_file_attached_on_the_continuation_reaches_the_model(self, tmp_path):
        attached = tmp_path / "loader.py"
        attached.write_text("MISSING_KEY_SENTINEL = 'fallback'\n")
        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        await _continue(tmp_path, thread_id, provider, files=[str(attached)])

        assert "MISSING_KEY_SENTINEL" in _sent_prompt(provider)

    @pytest.mark.asyncio
    async def test_directory_attached_on_the_continuation_reaches_the_model(self, tmp_path):
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "inner.py").write_text("DIRECTORY_SENTINEL = 1\n")
        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        await _continue(tmp_path, thread_id, provider, files=[str(package)])

        # Once: the history embeds the directory's files, so the tool must not
        # embed them again under the unexpanded directory path.
        assert _sent_prompt(provider).count("DIRECTORY_SENTINEL") == 1

    @pytest.mark.asyncio
    async def test_prompt_file_attached_on_the_continuation_reaches_the_model(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("PROMPT_FILE_SENTINEL: explain the fallback")
        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        await _continue(tmp_path, thread_id, provider, prompt="", files=[str(prompt_file)])

        prompt = _sent_prompt(provider)
        assert "PROMPT_FILE_SENTINEL" in prompt
        assert FIRST_REPLY in prompt
        # The request fills the NEW USER INPUT slot, ahead of the follow-up instructions.
        request_at = prompt.index("PROMPT_FILE_SENTINEL")
        assert prompt.index(NEW_USER_INPUT_MARKER) < request_at < prompt.index("CONVERSATION CONTINUATION:")

    @pytest.mark.asyncio
    async def test_oversize_prompt_on_the_continuation_is_refused(self, tmp_path):
        from config import MCP_PROMPT_SIZE_LIMIT

        thread_id = _chat_thread(tmp_path)
        provider = create_mock_provider(model_name="flash")

        with pytest.raises(ToolExecutionError) as refused:
            await _continue(tmp_path, thread_id, provider, prompt="x" * (MCP_PROMPT_SIZE_LIMIT + 1))

        assert json.loads(refused.value.payload)["status"] == "resend_prompt"
        provider.generate_content.assert_not_called()


class TestServerContinuationReloadsTheThread:
    @pytest.mark.asyncio
    async def test_follow_up_budget_counts_the_turns_before_this_one(self, tmp_path):
        thread_id = _chat_thread(tmp_path)

        with patch("server.get_follow_up_instructions", return_value="") as follow_up:
            await reconstruct_thread_context({"prompt": FOLLOW_UP, "continuation_id": thread_id, "model": "flash"})

        follow_up.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_storage_fault_on_the_reload_falls_back_to_the_loaded_thread(self, tmp_path):
        import utils.conversation_memory as cm

        thread_id = _chat_thread(tmp_path)
        real_get_thread = cm.get_thread

        def get_thread_failing_after_the_user_turn(tid):
            context = real_get_thread(tid)
            if context is not None and len(context.turns) > 2:
                raise cm.ConversationMemoryStorageError("backend down")
            return context

        with patch.object(cm, "get_thread", side_effect=get_thread_failing_after_the_user_turn):
            enhanced = await reconstruct_thread_context(
                {"prompt": FOLLOW_UP, "continuation_id": thread_id, "model": "flash"}
            )

        assert FIRST_REPLY in enhanced["prompt"]
        assert FOLLOW_UP in enhanced["prompt"]
