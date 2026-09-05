"""
Workflow metadata names a model only when the request actually called one.

Pins issue #96: ``codereview`` with ``review_validation_type="internal"`` used
to stamp ``metadata.model_used``/``provider_used`` with the requested model
even though no request ever reached it, and echoed the caller's own
``findings`` back as ``final_analysis`` with a hard-coded ``confidence_level``.

The external-mode case is the control: with a mocked provider the expert
branch runs, ``generate_content`` is awaited once, and the metadata names that
model. The other cases assert that ``generate_content`` was never awaited, so
``model_used is None`` is tied to an observed absence of a call rather than to
a status string.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from tests.mock_helpers import create_mock_provider
from tools.codereview import CodeReviewTool
from utils.model_context import ModelContext

MODEL_NAME = "flash"


def _server_arguments(model_context: ModelContext, **overrides):
    """Arguments as server.py hands them to a workflow tool (model already resolved)."""
    arguments = {
        "step": "Review the auth helper",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "Probe only. The file under review is a small TypeScript auth helper.",
        "model": MODEL_NAME,
        "_model_context": model_context,
        "_resolved_model_name": MODEL_NAME,
    }
    arguments.update(overrides)
    return arguments


@pytest.fixture
def reviewed_file(tmp_path):
    """Codereview step 1 requires relevant_files; give it a real absolute path."""
    path = tmp_path / "auth.ts"
    path.write_text("export const isAdmin = (u: { role: string }) => u.role === 'admin';\n")
    return str(path)


@pytest.fixture
def mocked_model_context():
    """A ModelContext whose provider is a mock, so the expert branch can run offline."""
    provider = create_mock_provider(model_name=MODEL_NAME)
    context = ModelContext(MODEL_NAME)
    context._provider = provider
    return context, provider


def _run(arguments):
    result = asyncio.run(CodeReviewTool().execute_workflow(arguments))
    assert len(result) == 1
    return json.loads(result[0].text)


class TestCodeReviewInternalMetadata:
    def test_internal_mode_final_step_names_no_model(self, reviewed_file, mocked_model_context):
        context, provider = mocked_model_context
        arguments = _server_arguments(
            context,
            review_validation_type="internal",
            relevant_files=[reviewed_file],
        )

        response = _run(arguments)

        provider.generate_content.assert_not_awaited()
        assert response["skip_expert_analysis"] is True
        assert response["expert_analysis"]["status"] == "skipped_due_to_internal_analysis_type"

        metadata = response["metadata"]
        assert metadata["tool_name"] == "codereview"
        assert metadata["model_used"] is None
        assert metadata["provider_used"] is None

        completion = response["complete_code_review"]
        assert completion["caller_findings"] == arguments["findings"]
        assert "final_analysis" not in completion
        assert "confidence_level" not in completion

    def test_internal_mode_echoes_findings_as_caller_findings(self, reviewed_file, mocked_model_context):
        """The payload shape alone, independent of the metadata assertion above."""
        context, _provider = mocked_model_context
        arguments = _server_arguments(
            context,
            review_validation_type="internal",
            relevant_files=[reviewed_file],
        )

        completion = _run(arguments)["complete_code_review"]

        assert completion["caller_findings"] == arguments["findings"]
        assert "final_analysis" not in completion, "the caller's own text must not be labelled as an analysis"
        assert "confidence_level" not in completion, "codereview has no caller-stated confidence to report"

    def test_external_mode_names_the_model_it_called(self, reviewed_file, mocked_model_context):
        context, provider = mocked_model_context
        arguments = _server_arguments(
            context,
            review_validation_type="external",
            relevant_files=[reviewed_file],
        )

        response = _run(arguments)

        provider.generate_content.assert_awaited_once()
        assert provider.generate_content.await_args.kwargs["model_name"] == MODEL_NAME
        assert response["expert_analysis"]["status"] == "analysis_complete"

        metadata = response["metadata"]
        assert metadata["tool_name"] == "codereview"
        assert metadata["model_used"] == MODEL_NAME
        assert metadata["provider_used"] == "google"

    def test_use_assistant_model_false_names_no_model(self, reviewed_file, mocked_model_context):
        context, provider = mocked_model_context
        arguments = _server_arguments(
            context,
            review_validation_type="external",
            use_assistant_model=False,
            relevant_files=[reviewed_file],
        )

        response = _run(arguments)

        provider.generate_content.assert_not_awaited()
        assert response["status"] == "local_work_complete"
        assert response["metadata"]["model_used"] is None
        assert response["metadata"]["provider_used"] is None

    def test_intermediate_step_names_no_model(self, reviewed_file, mocked_model_context):
        context, provider = mocked_model_context
        arguments = _server_arguments(
            context,
            total_steps=3,
            next_step_required=True,
            relevant_files=[reviewed_file],
        )

        response = _run(arguments)

        provider.generate_content.assert_not_awaited()
        # codereview renames the generic pause status (tools/codereview.py customize_workflow_response)
        assert response["status"] == "pause_for_code_review"
        assert response["metadata"]["model_used"] is None
        assert response["metadata"]["provider_used"] is None

    def test_external_mode_unresolved_provider_names_no_model(self, reviewed_file):
        """The expert branch is entered but no provider resolves, so no model is ever called.

        Nothing is injected into this ModelContext, so ``.provider`` asks the registry
        for a model it does not know and raises; _call_expert_analysis reports
        analysis_error and the workflow surfaces status "error". The audit fields must
        not name the model that was merely requested.
        """
        unknown_model = "no-such-model-xyz"
        arguments = _server_arguments(
            ModelContext(unknown_model),
            model=unknown_model,
            _resolved_model_name=unknown_model,
            review_validation_type="external",
            relevant_files=[reviewed_file],
        )

        response = _run(arguments)

        assert response["status"] == "error"
        assert unknown_model in response["content"]

        metadata = response["metadata"]
        assert metadata["tool_name"] == "codereview"
        assert metadata["model_used"] is None, "no request reached a model, so none may be named"
        assert metadata["provider_used"] is None

    def test_external_mode_model_call_error_still_names_the_model(self, reviewed_file, mocked_model_context):
        """The request reached the provider and the error came back from it, so the audit fields name it."""
        context, provider = mocked_model_context
        provider.generate_content = AsyncMock(side_effect=RuntimeError("upstream 503"))
        arguments = _server_arguments(context, review_validation_type="external", relevant_files=[reviewed_file])

        response = _run(arguments)

        provider.generate_content.assert_awaited_once()
        assert response["status"] == "error"
        assert "upstream 503" in response["content"]
        assert response["metadata"]["model_used"] == MODEL_NAME
        assert response["metadata"]["provider_used"] == "google"

    def test_reused_arguments_dict_does_not_carry_the_previous_calls_model(self, reviewed_file, mocked_model_context):
        """The record is per call: a second call through the same dict on a no-model path reports None."""
        context, _provider = mocked_model_context
        arguments = _server_arguments(context, review_validation_type="external", relevant_files=[reviewed_file])
        tool = CodeReviewTool()

        first = json.loads(asyncio.run(tool.execute_workflow(arguments))[0].text)
        assert first["metadata"]["model_used"] == MODEL_NAME

        arguments["review_validation_type"] = "internal"
        second = json.loads(asyncio.run(tool.execute_workflow(arguments))[0].text)

        assert second["skip_expert_analysis"] is True
        assert second["metadata"]["model_used"] is None
        assert second["metadata"]["provider_used"] is None
