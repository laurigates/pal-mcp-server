"""A workflow's final step counts files carried in from conversation history (issue #178).

The final step skips files that are already in the thread's history, so
``file_context.files_embedded`` used to be the number embedded in that step
alone: 0 on a two-step review whose files were all named in step 1, printed next
to "Full file content embedded for expert analysis". The calling agent reads the
field to judge whether the expert saw the code.
"""

import asyncio
import json

import pytest

from tests.mock_helpers import create_mock_provider
from tools.codereview import CodeReviewTool
from utils.model_context import ModelContext


@pytest.fixture
def reviewed_file(tmp_path):
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


def _arguments(**overrides):
    context = ModelContext("flash")
    context._provider = create_mock_provider(model_name="flash")
    arguments = {
        "step": "Review the auth helper",
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


def _run(arguments):
    result = asyncio.run(CodeReviewTool().execute_workflow(arguments))
    return json.loads(result[0].text)


def test_final_step_counts_files_already_in_history(reviewed_file):
    step1 = _run(_arguments(relevant_files=[reviewed_file], total_steps=2, next_step_required=True))

    step2 = _run(
        _arguments(
            relevant_files=[reviewed_file],
            step_number=2,
            total_steps=2,
            next_step_required=False,
            continuation_id=step1["continuation_id"],
        )
    )

    file_context = step2["file_context"]
    assert file_context["type"] == "fully_embedded"
    assert file_context["files_embedded_this_step"] == 0
    assert file_context["files_in_history"] == 1
    assert file_context["files_embedded"] == 1


def test_single_step_counts_its_own_files(reviewed_file):
    response = _run(_arguments(relevant_files=[reviewed_file]))

    file_context = response["file_context"]
    assert file_context["files_embedded_this_step"] == 1
    assert file_context["files_in_history"] == 0
    assert file_context["files_embedded"] == 1
