"""What a tool records as a conversation turn (issue #174).

The stored turns are replayed to the next model on continuation and shown in
transcripts and `pal://threads/<id>`. They used to depend on the tool and on
whether the thread was new:

* `chat` stored its formatted output, ending in the `AGENT'S TURN:` instruction
  meant for the calling agent, so the next model read PAL's instruction as if a
  previous model had written it.
* other simple tools stored formatted output on a new thread and raw model text
  on a continuation.
* workflow tools never stored the caller's `step` as a user turn, and their
  assistant turns carried no `model_name`/`model_provider` even when an expert
  model ran.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from providers.shared import ProviderType
from server import _record_user_turn_or_raise
from tests.mock_helpers import create_mock_provider
from tools.chat import ChatTool
from tools.codereview import CodeReviewTool
from utils.conversation_memory import create_thread, get_thread
from utils.model_context import ModelContext

FIRST_REPLY = "PONG"
SECOND_REPLY = "Still PONG."


class _ScriptedProvider:
    def __init__(self, *replies: str):
        self._replies = list(replies)

    def get_provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    async def generate_content(self, **kwargs):  # noqa: ARG002
        return SimpleNamespace(content=self._replies.pop(0), usage=None, metadata={})


def _model_context(provider):
    capabilities = SimpleNamespace(
        supports_extended_thinking=False,
        allow_code_generation=False,
        supports_images=False,
        temperature_constraint=None,
        get_effective_max_output_tokens=lambda: None,
    )
    return SimpleNamespace(
        model_name="gemini-2.5-flash",
        provider=provider,
        capabilities=capabilities,
        calculate_token_allocation=lambda: SimpleNamespace(file_tokens=50_000, history_tokens=50_000),
        estimate_tokens=lambda text: len(text) // 4,
    )


async def _chat(tool, provider, tmp_path, prompt, continuation_id=None):
    arguments = {
        "prompt": prompt,
        "model": "gemini-2.5-flash",
        "working_directory_absolute_path": str(tmp_path),
        "_model_context": _model_context(provider),
    }
    if continuation_id:
        arguments["continuation_id"] = continuation_id
    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        result = await tool.execute(arguments)
    return json.loads(result[0].text)


def _assistant_turns(thread_id):
    return [turn for turn in get_thread(thread_id).turns if turn.role == "assistant"]


class TestSimpleToolRecordsTheModelReply:
    @pytest.mark.asyncio
    async def test_new_and_continued_threads_record_the_reply_verbatim(self, tmp_path):
        tool = ChatTool()
        provider = _ScriptedProvider(FIRST_REPLY, SECOND_REPLY)

        first = await _chat(tool, provider, tmp_path, "Reply with PONG.")
        thread_id = first["continuation_offer"]["continuation_id"]
        await _chat(tool, provider, tmp_path, "Again?", continuation_id=thread_id)

        assert [turn.content for turn in _assistant_turns(thread_id)] == [FIRST_REPLY, SECOND_REPLY]

    @pytest.mark.asyncio
    async def test_the_agent_instruction_reaches_the_caller_but_not_the_thread(self, tmp_path):
        tool = ChatTool()
        provider = _ScriptedProvider(FIRST_REPLY)

        response = await _chat(tool, provider, tmp_path, "Reply with PONG.")
        thread_id = response["continuation_offer"]["continuation_id"]

        assert "AGENT'S TURN:" in response["content"]
        assert all("AGENT'S TURN:" not in turn.content for turn in get_thread(thread_id).turns)

    @pytest.mark.asyncio
    async def test_assistant_turn_names_the_model(self, tmp_path):
        tool = ChatTool()
        provider = _ScriptedProvider(FIRST_REPLY)

        response = await _chat(tool, provider, tmp_path, "Reply with PONG.")

        (turn,) = _assistant_turns(response["continuation_offer"]["continuation_id"])
        assert (turn.model_name, turn.model_provider) == ("gemini-2.5-flash", "google")


STEP_TEXT = "Review the auth helper for privilege checks"


@pytest.fixture
def reviewed_file(tmp_path):
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


def _codereview_arguments(**overrides):
    provider = create_mock_provider(model_name="flash")
    context = ModelContext("flash")
    context._provider = provider
    arguments = {
        "step": STEP_TEXT,
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "Probe only.",
        "model": "flash",
        "review_validation_type": "external",
        "_model_context": context,
        "_resolved_model_name": "flash",
    }
    arguments.update(overrides)
    return arguments


def _run_codereview(arguments):
    result = asyncio.run(CodeReviewTool().execute_workflow(arguments))
    return json.loads(result[0].text)


class TestWorkflowRecordsRequestAndModel:
    def test_new_thread_records_the_step_as_a_user_turn(self, reviewed_file):
        response = _run_codereview(_codereview_arguments(relevant_files=[reviewed_file]))

        turns = get_thread(response["continuation_id"]).turns
        assert [turn.role for turn in turns] == ["user", "assistant"]
        assert turns[0].content == STEP_TEXT
        assert turns[0].tool_name == "codereview"

    def test_expert_turn_names_the_model_and_provider(self, reviewed_file):
        response = _run_codereview(_codereview_arguments(relevant_files=[reviewed_file]))

        (turn,) = _assistant_turns(response["continuation_id"])
        assert (turn.model_name, turn.model_provider) == ("flash", "google")

    def test_step_without_a_model_call_names_no_model(self, reviewed_file):
        response = _run_codereview(
            _codereview_arguments(relevant_files=[reviewed_file], total_steps=2, next_step_required=True)
        )

        (turn,) = _assistant_turns(response["continuation_id"])
        assert (turn.model_name, turn.model_provider) == (None, None)

    def test_assistant_turn_carries_no_empty_step_field(self, reviewed_file):
        response = _run_codereview(_codereview_arguments(relevant_files=[reviewed_file]))

        (turn,) = _assistant_turns(response["continuation_id"])
        assert "step" not in json.loads(turn.content)["step_info"]

    def test_continuation_through_the_server_records_the_step(self):
        thread_id = create_thread("codereview", {"step": "first"})
        context = get_thread(thread_id)

        _record_user_turn_or_raise(thread_id, context, {"step": STEP_TEXT, "continuation_id": thread_id})

        (turn,) = get_thread(thread_id).turns
        assert (turn.role, turn.content) == ("user", STEP_TEXT)

    def test_prompt_still_wins_over_step(self):
        thread_id = create_thread("chat", {"prompt": "first"})
        context = get_thread(thread_id)

        _record_user_turn_or_raise(thread_id, context, {"prompt": "the prompt", "step": "a step"})

        (turn,) = get_thread(thread_id).turns
        assert turn.content == "the prompt"
