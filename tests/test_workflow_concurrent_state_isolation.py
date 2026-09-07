"""
Concurrent calls must not share workflow state (issue #99), and a continuation
must restore its tool config from the thread (issue #100).

These two are one change. server.py used to dispatch every request to a single
tool object per tool, and ``_call_expert_analysis`` awaits a model inside
``execute_workflow`` — so a second call arriving during that await reset and
refilled the same ``self``, and the first call resumed reading the second's
investigation. #97's per-call reset changed the shape of that (A's history is
replaced by B's rather than merged with it) without removing it.

Giving each call its own instance closes that, but it also removes the accident
that made continuations work: tool-specific step-1 config (codereview's
review_config and its siblings) used to survive only because the next step
happened to land on the same object. So the roster of state that has to live in
the thread is what these tests pin, from both ends.

Both tests drive ``server.handle_call_tool`` rather than ``tool.execute``. The
isolation fix is at the dispatch site, so a test that constructs one tool and
calls it twice would bypass the thing under test and pass either way.
"""

import asyncio
import json

import pytest

from server import handle_call_tool
from tools.workflow.workflow_mixin import BaseWorkflowMixin
from utils.conversation_memory import get_thread

EXPERT_AWAIT_SECONDS = 0.05
SECOND_CALL_DELAY_SECONDS = 0.01


def _codereview_args(path: str, findings: str, **overrides) -> dict:
    """A single-step external review, so the call reaches the expert-analysis await."""
    args = {
        "step": f"Review {path}",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": findings,
        "files_checked": [path],
        "relevant_files": [path],
        "model": "gemini-2.5-flash",
    }
    args.update(overrides)
    return args


def _parse(result) -> dict:
    assert len(result) == 1
    return json.loads(result[0].text)


def _stub_expert_analysis(monkeypatch, *, seconds: float = EXPERT_AWAIT_SECONDS):
    """Hold every call inside the expert await long enough for another to interleave."""

    async def _slow_expert(self, arguments, request):
        await asyncio.sleep(seconds)
        return {"status": "analysis_complete", "raw_analysis": "stubbed"}

    monkeypatch.setattr(BaseWorkflowMixin, "_call_expert_analysis", _slow_expert)


@pytest.mark.asyncio
async def test_concurrent_calls_report_only_their_own_investigation(tmp_path, monkeypatch):
    """Call B starting inside call A's expert await must not appear in A's response."""
    _stub_expert_analysis(monkeypatch)

    file_a = tmp_path / "alpha.py"
    file_a.write_text("# alpha\n")
    file_b = tmp_path / "beta.py"
    file_b.write_text("# beta\n")

    args_a = _codereview_args(str(file_a), "A-FINDING: alpha mishandles the tenant id")
    args_b = _codereview_args(str(file_b), "B-FINDING: beta leaks a file handle")

    async def _call_b_during_a_await():
        await asyncio.sleep(SECOND_CALL_DELAY_SECONDS)
        return await handle_call_tool("codereview", args_b)

    result_a, result_b = await asyncio.gather(
        handle_call_tool("codereview", args_a),
        _call_b_during_a_await(),
    )
    response_a, response_b = _parse(result_a), _parse(result_b)

    serialized_a, serialized_b = json.dumps(response_a), json.dumps(response_b)

    # Each response describes its own file and findings, and nothing of the other's.
    assert "alpha.py" in serialized_a
    assert "beta.py" not in serialized_a, "call A's response carries call B's file — shared state across the await"
    assert "B-FINDING" not in serialized_a

    assert "beta.py" in serialized_b
    assert "alpha.py" not in serialized_b, "call B's response carries call A's file — shared state across the await"
    assert "A-FINDING" not in serialized_b


@pytest.mark.asyncio
async def test_concurrent_calls_persist_only_their_own_thread(tmp_path, monkeypatch):
    """Each thread's stored work_history must describe only the call that created it."""
    _stub_expert_analysis(monkeypatch)

    file_a = tmp_path / "gamma.py"
    file_a.write_text("# gamma\n")
    file_b = tmp_path / "delta.py"
    file_b.write_text("# delta\n")

    args_a = _codereview_args(str(file_a), "A-PERSIST: gamma retries without a ceiling")
    args_b = _codereview_args(str(file_b), "B-PERSIST: delta swallows the timeout")

    async def _call_b_during_a_await():
        await asyncio.sleep(SECOND_CALL_DELAY_SECONDS)
        return await handle_call_tool("codereview", args_b)

    result_a, result_b = await asyncio.gather(
        handle_call_tool("codereview", args_a),
        _call_b_during_a_await(),
    )
    response_a, response_b = _parse(result_a), _parse(result_b)

    thread_a = get_thread(response_a["continuation_id"])
    thread_b = get_thread(response_b["continuation_id"])
    assert thread_a is not None and thread_b is not None
    assert response_a["continuation_id"] != response_b["continuation_id"]

    def _stored_work_history(thread):
        for turn in reversed(thread.turns):
            if turn.role == "assistant" and turn.tool_name == "codereview" and turn.model_metadata:
                return json.dumps(turn.model_metadata.get("work_history", []))
        raise AssertionError("no codereview state was persisted on the thread")

    history_a = _stored_work_history(thread_a)
    history_b = _stored_work_history(thread_b)

    assert "A-PERSIST" in history_a
    assert "B-PERSIST" not in history_a, "thread A persisted call B's findings"
    assert "B-PERSIST" in history_b
    assert "A-PERSIST" not in history_b, "thread B persisted call A's findings"


@pytest.mark.asyncio
async def test_continuation_restores_tool_config_from_the_thread(tmp_path, monkeypatch):
    """review_config is set on step 1 and must come back from the thread on step 2.

    With per-call instances there is no in-memory carry-over left to supply it,
    so this is what stops the expert prompt on a later step from running with an
    empty (or another caller's) review configuration.
    """
    _stub_expert_analysis(monkeypatch, seconds=0)

    target = tmp_path / "epsilon.py"
    target.write_text("# epsilon\n")

    step_one = await handle_call_tool(
        "codereview",
        _codereview_args(
            str(target),
            "Step one maps the module",
            total_steps=2,
            next_step_required=True,
            review_type="security",
            focus_on="authentication boundaries",
        ),
    )
    response_one = _parse(step_one)
    continuation_id = response_one["continuation_id"]

    thread = get_thread(continuation_id)
    stored = next(
        turn.model_metadata
        for turn in reversed(thread.turns)
        if turn.role == "assistant" and turn.tool_name == "codereview" and turn.model_metadata
    )
    tool_state = stored.get("tool_state") or {}
    review_config = tool_state.get("review_config") or {}

    # The config reached the thread at all — the half that was missing before #100.
    assert review_config.get("review_type") == "security"
    assert review_config.get("focus_on") == "authentication boundaries"

    # And a later step, which lands on a fresh instance, reads it back.
    from tools.codereview import CodeReviewTool

    resumed = CodeReviewTool()
    assert resumed.review_config == {}, "a new instance must start with no configuration"

    restored_state = resumed.load_persisted_workflow_state(continuation_id)
    assert restored_state is not None
    resumed.restore_tool_state(restored_state.get("tool_state") or {})

    assert resumed.review_config.get("review_type") == "security"
    assert resumed.review_config.get("focus_on") == "authentication boundaries"


def test_every_workflow_tool_with_step_one_config_declares_it():
    """The roster of persisted state must not drift as tools gain configuration.

    Each of these writes config on step 1 and reads it on later steps, so each
    must either list the attribute in PERSISTED_STATE_ATTRS or implement the
    hook pair directly (consensus does, because its state is lists not dicts).
    """
    from tools.analyze import AnalyzeTool
    from tools.codereview import CodeReviewTool
    from tools.consensus import ConsensusTool
    from tools.planner import PlannerTool
    from tools.precommit import PrecommitTool
    from tools.refactor import RefactorTool
    from tools.thinkdeep import ThinkDeepTool
    from tools.tracer import TracerTool

    expected = {
        CodeReviewTool: ("review_config",),
        AnalyzeTool: ("analysis_config",),
        PrecommitTool: ("git_config",),
        RefactorTool: ("refactor_config",),
        TracerTool: ("trace_config",),
        ThinkDeepTool: ("stored_request_params",),
        PlannerTool: ("branches",),
    }
    for tool_cls, attrs in expected.items():
        assert tool_cls.PERSISTED_STATE_ATTRS == attrs, (
            f"{tool_cls.__name__} no longer declares {attrs}; its step-1 config would "
            "stop being persisted and a continuation would silently run without it."
        )
        # The declared names must be real attributes, or persistence stores nothing.
        instance = tool_cls()
        for attr in attrs:
            assert hasattr(instance, attr), f"{tool_cls.__name__}.{attr} does not exist"

    # Consensus overrides the hooks instead; assert it actually round-trips.
    consensus = ConsensusTool()
    consensus.models_to_consult = [{"model": "gemini-2.5-flash", "stance": "for"}]
    consensus.accumulated_responses = [{"model": "gemini-2.5-flash", "status": "success"}]
    persisted = consensus.get_persisted_tool_state()

    revived = ConsensusTool()
    revived.restore_tool_state(persisted)
    assert revived.models_to_consult == consensus.models_to_consult
    assert revived.accumulated_responses == consensus.accumulated_responses
