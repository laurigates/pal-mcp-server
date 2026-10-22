"""file_context reports what the expert was sent (issue #189).

Every workflow reads its files on the final step, but only tools whose
``should_include_files_in_expert_prompt()`` is true put that content in the
expert prompt. Debug and thinkdeep keep the default (false), yet their final
step reported ``fully_embedded`` / "Full file content embedded for expert
analysis" for content the expert never received.
"""

import asyncio
import json

import pytest

from tests.mock_helpers import create_mock_provider
from tools.codereview import CodeReviewTool
from tools.debug import DebugIssueTool
from utils.model_context import ModelContext


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


def _arguments(source_file, **overrides):
    context = ModelContext("flash")
    context._provider = create_mock_provider(model_name="flash")
    arguments = {
        "step": "Investigate the auth helper",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "Probe only.",
        "relevant_files": [source_file],
        "model": "flash",
        "_model_context": context,
        "_resolved_model_name": "flash",
    }
    arguments.update(overrides)
    return arguments


def _run(tool, arguments):
    result = asyncio.run(tool.execute_workflow(arguments))
    return json.loads(result[0].text)


def test_debug_final_step_does_not_claim_files_reached_expert(source_file):
    tool = DebugIssueTool()
    assert tool.should_include_files_in_expert_prompt() is False

    response = _run(tool, _arguments(source_file, hypothesis="Role check is case-sensitive", confidence="high"))

    assert response["status"] == "calling_expert_analysis"
    file_context = response["file_context"]
    assert file_context["type"] == "not_sent_to_expert"
    assert file_context["files_embedded"] == 0
    assert file_context["files_read_this_step"] == 1
    assert file_context["files_in_history"] == 0
    assert "Full file content embedded for expert analysis" not in file_context["context_optimization"]
    assert "not included in the expert prompt" in file_context["context_optimization"]


def test_not_sent_to_expert_keeps_history_files_apart_from_this_steps_reads(source_file):
    """Files an earlier turn carries are skipped on this step, not read."""
    tool = DebugIssueTool()
    tool._actually_processed_files = [source_file]
    tool._files_in_history = ["/repo/earlier_a.py", "/repo/earlier_b.py"]

    assert tool._embedded_file_context() == {
        "type": "not_sent_to_expert",
        "files_embedded": 0,
        "files_read_this_step": 1,
        "files_in_history": 2,
        "context_optimization": "File content was not included in the expert prompt",
    }


def test_codereview_final_step_still_reports_fully_embedded(source_file):
    tool = CodeReviewTool()
    assert tool.should_include_files_in_expert_prompt() is True

    response = _run(tool, _arguments(source_file, review_validation_type="external"))

    assert response["file_context"] == {
        "type": "fully_embedded",
        "files_embedded": 1,
        "files_embedded_this_step": 1,
        "files_in_history": 0,
        "context_optimization": "Full file content embedded for expert analysis",
    }
