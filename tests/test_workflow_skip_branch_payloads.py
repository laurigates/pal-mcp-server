"""
Skip-branch completion payloads after issue #96, for the tools whose rule differs.

The completion builder (tools/workflow/base.py handle_completion_without_expert_analysis)
returns the caller's text as ``caller_findings`` and emits ``confidence_level`` only
when ``get_confidence_level`` returns a value. Two contracts must hold at once:

* debug's "certain" path keeps ``confidence_level == "certain"`` — the value comes
  from the request, and simulator_tests/test_debug_certain_confidence.py relies on it —
  while the echoed hypothesis is labelled ``caller_findings``, not ``final_analysis``;
* precommit's internal mode omits ``confidence_level`` altogether, because
  PrecommitRequest only inherits a confidence field the tool never uses
  (prepare_step_data overwrites it with a dummy "high"; the skip gate keys on
  precommit_type).

Neither path calls a model, so ``metadata.model_used`` is None in both.
"""

import asyncio
import json

from tools.debug import DebugIssueTool
from tools.precommit import PrecommitTool


def _run(tool, arguments: dict) -> dict:
    result = asyncio.run(tool.execute_workflow(arguments))
    assert len(result) == 1
    return json.loads(result[0].text)


class TestDebugCertainPath:
    def test_certain_final_step_keeps_confidence_and_labels_hypothesis_as_caller_findings(self):
        hypothesis = "The cache key omits the tenant id, so tenants read each other's entries."
        arguments = {
            "step": "Trace the cache key construction",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "cache_key() joins only (resource, id); tenant is never part of it.",
            "confidence": "certain",
            "hypothesis": hypothesis,
        }

        response = _run(DebugIssueTool(), arguments)

        assert response["status"] == "certain_confidence_proceed_with_fix"
        assert response["skip_expert_analysis"] is True

        completion = response["complete_investigation"]
        assert completion["caller_findings"] == hypothesis
        assert "final_analysis" not in completion, "the caller's own hypothesis must not be labelled as an analysis"
        assert completion["confidence_level"] == "certain", "debug's confidence comes from the request and stays"

        assert response["metadata"]["tool_name"] == "debug"
        assert response["metadata"]["model_used"] is None
        assert response["metadata"]["provider_used"] is None


class TestPrecommitInternalMode:
    def test_internal_final_step_omits_confidence_and_names_no_model(self, tmp_path):
        findings = "Diff touches only the README; no code paths changed."
        arguments = {
            "step": "Validate the staged README change",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": findings,
            "precommit_type": "internal",
            "path": str(tmp_path),  # step 1 requires the repository root
        }

        response = _run(PrecommitTool(), arguments)

        assert response["status"] == "validation_complete_ready_for_commit"
        assert response["skip_expert_analysis"] is True

        completion = response["complete_validation"]
        assert completion["caller_findings"] == findings
        assert "final_analysis" not in completion
        assert "confidence_level" not in completion, "precommit has no caller-stated confidence to report"

        assert response["metadata"]["tool_name"] == "precommit"
        assert response["metadata"]["model_used"] is None
        assert response["metadata"]["provider_used"] is None
