"""A new thread's header and first user turn reach the transcript before the model call (issue #177).

`tail -f` on a transcript is how `docs/logging.md` says to follow what a
delegated model is asked. A simple tool used to create the thread and add the
user turn only after the model replied, so for a first request nothing appeared
until the reply did, the user turn's timestamp recorded when the reply was
saved, and a failed call left no transcript at all.

These tests read the transcript from inside the provider call, which is the
moment a `tail -f` reader would be looking at it.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from providers.shared import ProviderType
from tools.chat import ChatTool
from tools.shared.exceptions import ToolExecutionError
from utils import transcripts

PROMPT = "What does the auth helper do?"


def _records_on_disk() -> dict[str, list[dict]]:
    directory = transcripts.get_transcripts_dir()
    if not directory.exists():
        return {}
    return {
        path.stem: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in directory.glob("*.jsonl")
    }


class _ObservingProvider:
    """Records what the transcript holds at the moment the model is called."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.seen_during_call: dict[str, list[dict]] | None = None

    def get_provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    async def generate_content(self, **kwargs):  # noqa: ARG002
        self.seen_during_call = _records_on_disk()
        if self.fail:
            raise RuntimeError("provider exploded")
        return SimpleNamespace(content="The helper checks the admin role.", usage=None, metadata={})


def _model_context(provider: _ObservingProvider):
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


async def _run_chat(provider: _ObservingProvider, tmp_path):
    tool = ChatTool()
    with patch.object(tool, "get_validated_temperature", return_value=(0.5, [])):
        return await tool.execute(
            {
                "prompt": PROMPT,
                "model": "gemini-2.5-flash",
                "working_directory_absolute_path": str(tmp_path),
                "_model_context": _model_context(provider),
            }
        )


@pytest.mark.asyncio
async def test_header_and_user_turn_are_on_disk_while_the_model_runs(tmp_path):
    provider = _ObservingProvider()

    await _run_chat(provider, tmp_path)

    assert provider.seen_during_call is not None
    assert len(provider.seen_during_call) == 1
    (records,) = provider.seen_during_call.values()
    assert [record["type"] for record in records] == ["thread", "turn"]
    assert records[1]["role"] == "user"
    assert records[1]["content"] == PROMPT


@pytest.mark.asyncio
async def test_reply_lands_in_the_thread_the_response_offers(tmp_path):
    provider = _ObservingProvider()

    result = await _run_chat(provider, tmp_path)

    payload = json.loads(result[0].text)
    thread_id = payload["continuation_offer"]["continuation_id"]
    (seen_id,) = provider.seen_during_call
    assert seen_id == thread_id

    records = _records_on_disk()[thread_id]
    assert [(r["type"], r.get("role")) for r in records] == [("thread", None), ("turn", "user"), ("turn", "assistant")]


@pytest.mark.asyncio
async def test_user_turn_timestamp_precedes_the_reply(tmp_path):
    provider = _ObservingProvider()

    result = await _run_chat(provider, tmp_path)

    thread_id = json.loads(result[0].text)["continuation_offer"]["continuation_id"]
    _, user_turn, assistant_turn = _records_on_disk()[thread_id]
    (seen_records,) = provider.seen_during_call.values()
    assert user_turn["timestamp"] == seen_records[1]["timestamp"]
    assert user_turn["timestamp"] < assistant_turn["timestamp"]


@pytest.mark.asyncio
async def test_failed_model_call_leaves_the_request_in_the_transcript(tmp_path):
    provider = _ObservingProvider(fail=True)

    with pytest.raises(ToolExecutionError):
        await _run_chat(provider, tmp_path)

    on_disk = _records_on_disk()
    assert len(on_disk) == 1
    (records,) = on_disk.values()
    assert [record["type"] for record in records] == ["thread", "turn"]
    assert records[1]["content"] == PROMPT
