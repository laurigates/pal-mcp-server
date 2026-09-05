"""
Per-call state isolation for workflow tools (issue #97).

server.py builds TOOLS once and dispatches every call to the same instance, so
anything a call leaves on ``self`` is visible to the next call. A call with no
continuation_id must therefore start from empty workflow state, and a call
with a continuation_id must rebuild its state from the stored thread rather
than from whatever the instance happened to do last.
"""

import json

import pytest

from tools.codereview import CodeReviewTool
from tools.debug import DebugIssueTool
from tools.planner import PlannerTool
from tools.thinkdeep import ThinkDeepTool
from tools.tracer import TracerTool


def _codereview_args(path: str, findings: str, *, validation_type: str | None = "internal", **overrides) -> dict:
    """``validation_type=None`` omits review_validation_type, which defaults to external."""
    args = {
        "step": f"Review {path}",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": findings,
        "files_checked": [path],
        "relevant_files": [path],
    }
    if validation_type is not None:
        args["review_validation_type"] = validation_type
    args.update(overrides)
    return args


def _tracer_args(step: str, trace_mode: str, target_description: str, **overrides) -> dict:
    args = {
        "step": step,
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": f"Traced {target_description}",
        "trace_mode": trace_mode,
        "target_description": target_description,
    }
    args.update(overrides)
    return args


async def _run(tool, arguments: dict) -> dict:
    result = await tool.execute(arguments)
    assert len(result) == 1
    return json.loads(result[0].text)


@pytest.mark.asyncio
async def test_fresh_call_does_not_inherit_previous_call_state(tmp_path):
    """Two calls with no continuation_id on ONE instance: the second response
    must describe only the second call (issue #97)."""
    first = tmp_path / "dirA" / "auth.ts"
    second = tmp_path / "dirB" / "billing.ts"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("export const x = 1;\n")

    tool = CodeReviewTool()
    await _run(tool, _codereview_args(str(first), "FIRST-CALL-FINDING"))
    response = await _run(tool, _codereview_args(str(second), "SECOND-CALL-FINDING"))

    review = response["complete_code_review"]
    assert review["steps_taken"] == 1
    assert review["files_examined"] == [str(second)]
    assert review["relevant_files"] == [str(second)]
    assert "SECOND-CALL-FINDING" in review["work_summary"]
    assert "FIRST-CALL-FINDING" not in review["work_summary"]
    assert response["code_review_status"]["files_checked"] == 1
    assert len(tool.work_history) == 1


@pytest.mark.asyncio
async def test_continuation_restores_state_from_stored_thread(tmp_path):
    """A continuation call rebuilds work_history from the thread. Step 2 runs
    on a fresh instance so the restore is proven to come from storage, not
    from state left on the instance that ran step 1."""
    path = tmp_path / "auth.ts"
    path.write_text("export const x = 1;\n")

    step1 = await _run(
        CodeReviewTool(),
        _codereview_args(str(path), "STEP-ONE-FINDING", total_steps=2, next_step_required=True),
    )
    continuation_id = step1["continuation_id"]

    tool = CodeReviewTool()
    response = await _run(
        tool,
        _codereview_args(
            str(path),
            "STEP-TWO-FINDING",
            step_number=2,
            total_steps=2,
            continuation_id=continuation_id,
        ),
    )

    review = response["complete_code_review"]
    assert review["steps_taken"] == 2
    assert "STEP-ONE-FINDING" in review["work_summary"]
    assert "STEP-TWO-FINDING" in review["work_summary"]
    assert [step["step_number"] for step in tool.work_history] == [1, 2]
    assert tool.initial_request == f"Review {path}"


@pytest.mark.asyncio
async def test_unknown_continuation_id_starts_from_empty_state():
    """A continuation_id that resolves to no thread (only reachable when the
    tool is called directly; server.py rejects it earlier) must not fall back
    to the previous call's in-memory state."""
    tool = PlannerTool()
    await tool.execute({"step": "First plan", "step_number": 1, "total_steps": 2, "next_step_required": True})
    await tool.execute(
        {
            "step": "Unrelated plan",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "continuation_id": "not-a-thread-id",
        }
    )

    assert [step["step"] for step in tool.work_history] == ["Unrelated plan"]


@pytest.mark.asyncio
async def test_continuation_refills_tool_specific_initial_issue():
    """debug reads ``initial_issue`` (its own slot, filled through
    store_initial_issue) rather than ``initial_request``. A continuation on a
    fresh instance must refill it from the thread, or the expert prompt falls
    back to the generic placeholder."""
    step1_args = {
        "step": "Investigate the flaky login timeout",
        "step_number": 1,
        "total_steps": 2,
        "next_step_required": True,
        "findings": "STEP-ONE-FINDING",
    }
    step1 = json.loads((await DebugIssueTool().execute(step1_args))[0].text)

    tool = DebugIssueTool()
    await tool.execute(
        {
            "step": "Check the retry path",
            "step_number": 2,
            "total_steps": 2,
            "next_step_required": True,
            "findings": "STEP-TWO-FINDING",
            "continuation_id": step1["continuation_id"],
        }
    )

    assert tool.initial_issue == "Investigate the flaky login timeout"
    assert [step["step_number"] for step in tool.work_history] == [1, 2]


@pytest.mark.asyncio
async def test_cross_tool_continuation_keeps_debug_issue_text():
    """debug step 1 carrying another tool's continuation_id (the chat/analyze
    into debug flow) restores nothing from the thread. The mixin must still
    record that step as the initial request, or the persisted state carries
    initial_request None and every later step's expert prompt falls back to
    the 'Investigation initiated' placeholder."""
    from utils.conversation_memory import add_turn, create_thread

    thread_id = create_thread("chat", {"prompt": "Why does login time out?"})
    assert add_turn(thread_id, "user", "Why does login time out?", tool_name="chat")

    issue_text = "ISSUE-TEXT login times out after the session refactor"
    step1 = json.loads(
        (
            await DebugIssueTool().execute(
                {
                    "step": issue_text,
                    "step_number": 1,
                    "total_steps": 2,
                    "next_step_required": True,
                    "findings": "STEP-ONE-FINDING",
                    "continuation_id": thread_id,
                }
            )
        )[0].text
    )
    assert step1["continuation_id"] == thread_id

    tool = DebugIssueTool()
    await tool.execute(
        {
            "step": "Check the retry path",
            "step_number": 2,
            "total_steps": 2,
            "next_step_required": True,
            "findings": "STEP-TWO-FINDING",
            "continuation_id": thread_id,
        }
    )

    assert tool.initial_issue == issue_text
    assert issue_text in tool.prepare_expert_analysis_context(tool.consolidated_findings)


@pytest.mark.asyncio
async def test_external_expert_prompt_carries_only_this_calls_review_config(tmp_path, monkeypatch):
    """The default (external) path: the expert prompt's review configuration
    must be this call's. review_config used to be written in
    customize_workflow_response, which runs after the expert call, so the
    prompt carried the previous call's paths and focus."""
    first = tmp_path / "dirA" / "auth.ts"
    second = tmp_path / "dirB" / "billing.ts"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("export const x = 1;\n")

    tool = CodeReviewTool()
    prompts: list[str] = []

    async def capture_prompt(arguments, request):
        prompts.append(tool.prepare_expert_analysis_context(tool.consolidated_findings))
        return {"status": "analysis_complete", "raw_analysis": "stub"}

    monkeypatch.setattr(tool, "_call_expert_analysis", capture_prompt)

    await _run(tool, _codereview_args(str(first), "FIRST-CALL-FINDING", validation_type=None, focus_on="FIRST-FOCUS"))
    await _run(
        tool, _codereview_args(str(second), "SECOND-CALL-FINDING", validation_type=None, focus_on="SECOND-FOCUS")
    )

    assert len(prompts) == 2
    assert "FIRST-FOCUS" not in prompts[1]
    assert str(first) not in prompts[1]
    assert "SECOND-FOCUS" in prompts[1]
    assert str(second) in prompts[1]


@pytest.mark.asyncio
async def test_thinkdeep_expert_parameters_are_this_calls_own(monkeypatch):
    """thinkdeep reads stored_request_params during the expert call. They used
    to be written in customize_workflow_response, after that call, so a caller
    who passed no thinking_mode or temperature got the previous caller's."""
    tool = ThinkDeepTool()
    recorded: list[tuple[str, float]] = []

    async def record_params(arguments, request):
        recorded.append((tool.get_request_thinking_mode(request), tool.get_request_temperature(request)))
        return {"status": "analysis_complete", "raw_analysis": "stub"}

    monkeypatch.setattr(tool, "_call_expert_analysis", record_params)

    base = {
        "step": "Weigh the two cache designs",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "Write-through is simpler; write-back is faster",
    }
    await _run(tool, {**base, "thinking_mode": "max", "temperature": 0.9})
    await _run(tool, base)

    assert recorded[0] == ("max", 0.9)
    assert recorded[1] == (tool.get_expert_thinking_mode(), tool.get_default_temperature())
    assert recorded[1][0] != "max"
    assert recorded[1][1] != 0.9


@pytest.mark.asyncio
async def test_tracer_response_carries_its_own_trace_config():
    """Two fresh tracer calls on one instance: the second response must
    describe the second call's trace mode and target. The second call is a
    first step in ask mode with more steps to come, because the step-1
    guidance (required_actions, next_steps) reads trace_config before
    customize_workflow_response runs; it used to carry the previous call's
    mode."""
    tool = TracerTool()
    await _run(tool, _tracer_args("Trace login()", "precision", "FIRST-TARGET login()"))
    response = await _run(
        tool,
        _tracer_args(
            "Map BillingService",
            "ask",
            "SECOND-TARGET BillingService",
            total_steps=3,
            next_step_required=True,
        ),
    )

    assert response["metadata"]["trace_mode"] == "ask"
    assert response["metadata"]["target_description"] == "SECOND-TARGET BillingService"
    assert response["status"] == "mode_selection_required"
    assert "ask user" in response["required_actions"][0].lower()
    assert "ask the user to choose a tracing mode" in response["next_steps"].lower()


@pytest.mark.asyncio
async def test_step_one_reissued_on_same_tool_thread_records_the_new_issue():
    """A step 1 that reuses a thread already holding this tool's state (the schema
    tells clients to always reuse the last continuation_id) starts a new
    investigation: its own step text must be the issue the expert prompt carries,
    not the earlier investigation's, which the thread restore would otherwise keep."""
    old_issue = "OLD-ISSUE first investigation"
    new_issue = "NEW-ISSUE second investigation"

    def _step(step: str, number: int, findings: str, **overrides) -> dict:
        args = {"step": step, "step_number": number, "total_steps": 2, "next_step_required": True, "findings": findings}
        args.update(overrides)
        return args

    first = json.loads((await DebugIssueTool().execute(_step(old_issue, 1, "OLD-FINDING")))[0].text)
    thread_id = first["continuation_id"]
    await DebugIssueTool().execute(_step("Check the first lead", 2, "OLD-FINDING-TWO", continuation_id=thread_id))

    await DebugIssueTool().execute(_step(new_issue, 1, "NEW-FINDING", continuation_id=thread_id))
    tool = DebugIssueTool()
    await tool.execute(_step("Check the new lead", 2, "NEW-FINDING-TWO", continuation_id=thread_id))

    assert tool.initial_issue == new_issue
    context = tool.prepare_expert_analysis_context(tool.consolidated_findings)
    assert new_issue in context
    assert old_issue not in context
