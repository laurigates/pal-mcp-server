"""Pin the removal of schema parameters no tool reads (issues #102, #103).

A parameter is only truly gone when it is absent from what the server
advertises, so every assertion here inspects the *emitted* ``get_input_schema()``
rather than the source that builds it.

Two defects, one shape — advertised, accepted, then discarded:

* ``use_assistant_model`` on ``planner``/``tracer``/``consensus``/``docgen``,
  whose ``requires_expert_analysis()`` is ``False`` (#103),
* a required ``model`` on ``planner``/``tracer``, whose ``requires_model()`` is
  ``False`` (#103), and
* ``confidence`` on ``codereview``/``precommit``, which ``prepare_step_data``
  overwrites with a dummy ``"high"`` while the skip gate keys on
  ``review_validation_type``/``precommit_type`` (#102).
"""

import pytest

from tools.codereview import CodeReviewTool
from tools.consensus import ConsensusTool
from tools.docgen import DocgenTool
from tools.planner import PlannerTool
from tools.precommit import PrecommitTool
from tools.tracer import TracerTool

# Tools whose requires_expert_analysis() is False: nothing consults use_assistant_model.
SELF_CONTAINED_TOOLS = {
    "planner": PlannerTool,
    "tracer": TracerTool,
    "consensus": ConsensusTool,
    "docgen": DocgenTool,
}

# Tools whose requires_model() is False: model must not be a required parameter.
MODELLESS_TOOLS = {
    "planner": PlannerTool,
    "tracer": TracerTool,
}

# Tools that overwrite confidence with a dummy value and never read the caller's.
CONFIDENCE_IGNORING_TOOLS = {
    "codereview": CodeReviewTool,
    "precommit": PrecommitTool,
}


class TestUseAssistantModelNotAdvertised:
    @pytest.mark.parametrize("name", sorted(SELF_CONTAINED_TOOLS))
    def test_tool_is_self_contained(self, name):
        """Control: the premise for removing the parameter still holds."""
        tool = SELF_CONTAINED_TOOLS[name]()
        assert tool.requires_expert_analysis() is False
        assert tool.should_call_expert_analysis(None) is False

    @pytest.mark.parametrize("name", sorted(SELF_CONTAINED_TOOLS))
    def test_emitted_schema_omits_use_assistant_model(self, name):
        schema = SELF_CONTAINED_TOOLS[name]().get_input_schema()
        assert "use_assistant_model" not in schema["properties"]
        assert "use_assistant_model" not in schema["required"]


class TestModelNotRequired:
    # Exercise the real auto-mode branch; conftest's mock_provider would stub it to False
    # and make the required-list assertion vacuous.
    pytestmark = pytest.mark.no_mock_provider

    @pytest.mark.parametrize("name", sorted(MODELLESS_TOOLS))
    def test_model_is_not_required_even_in_auto_mode(self, name, monkeypatch):
        """Auto mode must not force a model on a tool that never calls one.

        ``DEFAULT_MODEL`` is pinned to ``auto`` so the assertion cannot pass
        vacuously on a machine where auto mode happens to be off.
        """
        import config

        monkeypatch.setattr(config, "DEFAULT_MODEL", "auto")
        tool = MODELLESS_TOOLS[name]()
        assert tool.requires_model() is False
        assert tool.is_effective_auto_mode() is True

        schema = tool.get_input_schema()
        assert "model" not in schema["required"]

    @pytest.mark.parametrize("name", sorted(MODELLESS_TOOLS))
    def test_model_is_still_accepted(self, name):
        """Existing callers that send a model must not start failing validation."""
        schema = MODELLESS_TOOLS[name]().get_input_schema()
        assert "model" in schema["properties"]


class TestConfidenceNotAdvertised:
    @pytest.mark.parametrize("name", sorted(CONFIDENCE_IGNORING_TOOLS))
    def test_tool_reports_no_confidence(self, name):
        """Control: the premise for removing the parameter still holds."""
        tool = CONFIDENCE_IGNORING_TOOLS[name]()
        assert tool.get_confidence_level(None) is None

    @pytest.mark.parametrize("name", sorted(CONFIDENCE_IGNORING_TOOLS))
    def test_emitted_schema_omits_confidence(self, name):
        schema = CONFIDENCE_IGNORING_TOOLS[name]().get_input_schema()
        assert "confidence" not in schema["properties"]
        assert "confidence" not in schema["required"]

    def test_codereview_request_no_longer_declares_a_deprecated_confidence(self):
        """The deprecated override is gone; the inherited field keeps old callers working."""
        from tools.codereview import CodeReviewRequest

        assert "confidence" not in CodeReviewRequest.__annotations__
        request = CodeReviewRequest(
            step="s",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="f",
            relevant_files=["/tmp/example.py"],
            confidence="high",
        )
        assert request.confidence == "high"


class TestNoCollateralSchemaLoss:
    """The shared builder edits must not narrow a tool that was never in scope."""

    def test_debug_still_advertises_confidence_and_use_assistant_model(self):
        from tools.debug import DebugIssueTool

        schema = DebugIssueTool().get_input_schema()
        assert "confidence" in schema["properties"]
        assert "use_assistant_model" in schema["properties"]

    def test_tracer_still_advertises_confidence(self):
        """Only use_assistant_model and the model requirement changed for tracer."""
        assert "confidence" in TracerTool().get_input_schema()["properties"]
