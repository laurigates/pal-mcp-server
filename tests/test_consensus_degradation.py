"""Tests that a consensus run reports models it lost instead of hiding them.

A model that fails at call time (retired model id, provider outage, bad key) is
recorded with status "error" and the workflow carries on to synthesis. Counting
those as responses reported a full-strength consensus that never happened.
"""

from tools.consensus import ConsensusTool


def _success(model: str, provider: str = "google", stance: str = "neutral") -> dict:
    return {
        "model": model,
        "stance": stance,
        "status": "success",
        "verdict": "looks fine",
        "metadata": {"provider": provider, "model_name": model},
    }


def _failure(model: str, error: str = "404 NOT_FOUND", stance: str = "neutral") -> dict:
    return {"model": model, "stance": stance, "status": "error", "error": error}


def _tool_with(responses: list[dict]) -> ConsensusTool:
    tool = ConsensusTool()
    tool.accumulated_responses = responses
    tool.original_proposal = "Should we ship it?"
    return tool


class TestFullConsensus:
    """A roster where every model answered is reported at full confidence."""

    def test_confidence_is_high(self):
        tool = _tool_with([_success("gemini-3.5-flash"), _success("gpt-5.2", provider="openai")])

        summary = tool._build_complete_consensus()

        assert summary["consensus_confidence"] == "high"

    def test_counts_match_the_roster(self):
        tool = _tool_with([_success("gemini-3.5-flash"), _success("gpt-5.2", provider="openai")])

        summary = tool._build_complete_consensus()

        assert summary["total_responses"] == 2
        assert summary["models_requested"] == 2

    def test_no_failure_keys_are_added(self):
        tool = _tool_with([_success("gemini-3.5-flash")])

        summary = tool._build_complete_consensus()

        assert "models_failed" not in summary

    def test_synthesis_has_no_warning(self):
        tool = _tool_with([_success("gemini-3.5-flash")])

        next_steps = tool._synthesis_next_steps(tool._build_complete_consensus())

        assert "CONSENSUS DEGRADED" not in next_steps
        assert next_steps.startswith("CONSENSUS GATHERING IS COMPLETE")


class TestDegradedConsensus:
    """A roster that lost a model must say so."""

    def _degraded(self) -> ConsensusTool:
        return _tool_with(
            [
                _success("gemini-3.5-flash", provider="google"),
                _failure("gemini-3-pro-preview", "404 NOT_FOUND - no longer available"),
                _success("gpt-5.2", provider="openai"),
            ]
        )

    def test_confidence_is_degraded(self):
        summary = self._degraded()._build_complete_consensus()

        assert summary["consensus_confidence"] == "degraded"

    def test_failed_model_is_excluded_from_the_response_count(self):
        summary = self._degraded()._build_complete_consensus()

        assert summary["total_responses"] == 2
        assert summary["models_requested"] == 3

    def test_failed_model_is_not_listed_as_consulted(self):
        summary = self._degraded()._build_complete_consensus()

        assert "gemini-3-pro-preview:neutral" not in summary["models_consulted"]

    def test_failure_is_reported_with_its_error(self):
        summary = self._degraded()._build_complete_consensus()

        assert len(summary["models_failed"]) == 1
        failure = summary["models_failed"][0]
        assert failure["model"] == "gemini-3-pro-preview"
        assert "404" in failure["error"]

    def test_surviving_provider_coverage_is_reported(self):
        summary = self._degraded()._build_complete_consensus()

        assert summary["providers_represented"] == ["google", "openai"]

    def test_synthesis_warns_and_names_the_lost_model(self):
        tool = self._degraded()

        next_steps = tool._synthesis_next_steps(tool._build_complete_consensus())

        assert "CONSENSUS DEGRADED" in next_steps
        assert "2 of 3" in next_steps
        assert "gemini-3-pro-preview" in next_steps
        # The synthesis instructions still follow the warning.
        assert "Key points of AGREEMENT" in next_steps


class TestProviderCoverageLoss:
    """Losing the only model from a provider drops that provider from coverage."""

    def test_lost_provider_is_absent_from_coverage(self):
        tool = _tool_with(
            [
                _success("gpt-5.2", provider="openai"),
                _failure("gemini-3-pro-preview"),
            ]
        )

        summary = tool._build_complete_consensus()

        assert summary["providers_represented"] == ["openai"]

    def test_every_model_failing_yields_no_responses(self):
        tool = _tool_with([_failure("gemini-3-pro-preview"), _failure("gpt-5.2")])

        summary = tool._build_complete_consensus()

        assert summary["total_responses"] == 0
        assert summary["consensus_confidence"] == "degraded"
        assert len(summary["models_failed"]) == 2
