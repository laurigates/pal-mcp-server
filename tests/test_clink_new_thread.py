"""A first clink call opens a thread and offers to continue it.

clink overrides ``execute`` and never reaches ``SimpleTool``'s model-call path,
so it has to open the thread itself. When the thread was moved ahead of the
model call (issue #177), clink was left calling ``_create_continuation_offer``
without a thread, and every first call came back ``success`` with no
``continuation_id``. These tests also read the transcript from inside the CLI
run, as the simple-tool tests do for the provider call.
"""

import json

import pytest

from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkTool
from utils import transcripts

PROMPT = "Summarize the project"
REPLY = "Hello from Gemini"


def _records_on_disk() -> dict[str, list[dict]]:
    directory = transcripts.get_transcripts_dir()
    if not directory.exists():
        return {}
    return {
        path.stem: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for path in directory.glob("*.jsonl")
    }


class _ObservingAgent:
    def __init__(self):
        self.seen_during_run: dict[str, list[dict]] | None = None

    async def run(self, **kwargs):  # noqa: ARG002
        self.seen_during_run = _records_on_disk()
        return AgentOutput(
            parsed=ParsedCLIResponse(content=REPLY, metadata={"model_used": "gemini-2.5-pro"}),
            sanitized_command=["gemini", "-o", "json"],
            returncode=0,
            stdout=json.dumps({"response": REPLY}),
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )


async def _run_clink(monkeypatch) -> tuple[dict, _ObservingAgent]:
    agent = _ObservingAgent()
    monkeypatch.setattr("tools.clink.create_agent", lambda client: agent)
    results = await CLinkTool().execute(
        {"prompt": PROMPT, "cli_name": "gemini", "role": "default", "absolute_file_paths": [], "images": []}
    )
    return json.loads(results[0].text), agent


@pytest.mark.asyncio
async def test_first_clink_call_offers_a_continuation(monkeypatch):
    payload, _ = await _run_clink(monkeypatch)

    assert payload["status"] == "continuation_available"
    assert payload["continuation_offer"]["continuation_id"]
    assert payload["metadata"]["cli_name"] == "gemini"


@pytest.mark.asyncio
async def test_offered_thread_holds_the_request_and_the_reply(monkeypatch):
    payload, agent = await _run_clink(monkeypatch)

    thread_id = payload["continuation_offer"]["continuation_id"]
    assert list(agent.seen_during_run) == [thread_id]
    assert [(r["type"], r.get("role")) for r in agent.seen_during_run[thread_id]] == [
        ("thread", None),
        ("turn", "user"),
    ]

    records = _records_on_disk()[thread_id]
    assert [(r["type"], r.get("role")) for r in records] == [("thread", None), ("turn", "user"), ("turn", "assistant")]
    assert records[1]["content"] == PROMPT
    assert REPLY in records[2]["content"]
