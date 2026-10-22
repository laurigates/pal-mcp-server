"""
Gemini 3.x thinking-level clamping is per-model, not one hardcoded mapping.

issue #137: providers/gemini.py used to apply a single THINKING_LEVELS mapping
to every Gemini 3.x model. That made thinking_mode="medium" a hard
INVALID_ARGUMENT on any model that only accepts low/high, and rounded
thinking_mode="minimal" up to LOW even on models that accept MINIMAL.

ModelCapabilities.supported_thinking_levels now records, per model, which
levels it accepts. Absence of the field (None) must reproduce today's
behaviour exactly -- that is the additive-ness guarantee, proven here by
calling ``_thinking_config`` with ``model_config=None`` and with a capability
object that leaves the field at its default.
"""

from google.genai import types

from providers.gemini import GeminiModelProvider
from providers.shared import ModelCapabilities, ProviderType


def _provider() -> GeminiModelProvider:
    return GeminiModelProvider(api_key="dummy-key-for-tests")


def _capabilities(model_name: str, **overrides) -> ModelCapabilities:
    return ModelCapabilities(
        provider=ProviderType.GOOGLE,
        model_name=model_name,
        friendly_name="Gemini (test)",
        supports_extended_thinking=True,
        max_thinking_tokens=24576,
        **overrides,
    )


class TestSupportedThinkingLevelsClamping:
    def test_low_high_only_model_gets_low_for_medium(self):
        """A model recording only low/high degrades medium to low rather than sending an unaccepted MEDIUM."""
        provider = _provider()
        config = provider._thinking_config(
            "gemini-3.8-flash",
            "medium",
            _capabilities("gemini-3.8-flash", supported_thinking_levels=["low", "high"]),
        )
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.LOW
        assert config.thinking_budget is None

    def test_model_recording_minimal_gets_minimal_not_low(self):
        """A model that records minimal as accepted receives MINIMAL, not the universal round-up to LOW."""
        provider = _provider()
        config = provider._thinking_config(
            "gemini-3.8-flash",
            "minimal",
            _capabilities(
                "gemini-3.8-flash",
                supported_thinking_levels=["minimal", "low", "medium", "high"],
            ),
        )
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.MINIMAL
        assert config.thinking_budget is None

    def test_high_only_model_degrades_medium_upward_when_nothing_lower_is_accepted(self):
        """With no lower accepted level available, clamping falls back upward instead of dropping thinking."""
        provider = _provider()
        config = provider._thinking_config(
            "gemini-3.8-flash",
            "medium",
            _capabilities("gemini-3.8-flash", supported_thinking_levels=["high"]),
        )
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.HIGH

    def test_unknown_mode_with_recorded_set_sends_no_thinking_config(self):
        provider = _provider()
        config = provider._thinking_config(
            "gemini-3.8-flash",
            "bogus",
            _capabilities("gemini-3.8-flash", supported_thinking_levels=["low", "high"]),
        )
        assert config is None


class TestNoRecordedSetIsUnchanged:
    """The additive-ness proof: a 3.x model with no supported_thinking_levels (the field's default,
    ``None``) behaves exactly as it did before this field existed."""

    def test_medium_still_maps_to_medium(self):
        provider = _provider()
        config = provider._thinking_config("gemini-3.6-flash", "medium", _capabilities("gemini-3.6-flash"))
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.MEDIUM
        assert config.thinking_budget is None

    def test_minimal_still_rounds_up_to_low(self):
        provider = _provider()
        config = provider._thinking_config("gemini-3.6-flash", "minimal", _capabilities("gemini-3.6-flash"))
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.LOW
        assert config.thinking_budget is None

    def test_max_still_maps_to_high(self):
        provider = _provider()
        config = provider._thinking_config("gemini-3.6-flash", "max", _capabilities("gemini-3.6-flash"))
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.HIGH

    def test_unknown_mode_still_sends_no_thinking_config(self):
        provider = _provider()
        assert provider._thinking_config("gemini-3.6-flash", "bogus", _capabilities("gemini-3.6-flash")) is None

    def test_no_model_config_at_all_still_works_like_today(self):
        """model_config can be None entirely (e.g. an unregistered model name); 3.x models don't need it."""
        provider = _provider()
        config = provider._thinking_config("gemini-3.6-flash", "medium", None)
        assert config is not None
        assert config.thinking_level == types.ThinkingLevel.MEDIUM


class TestGemini4ArgonRegistry:
    def test_argon_is_promoted_with_vendor_limits(self):
        """Gemini 4 Argon carries the limits from Google's model page and owns the ``pro`` alias."""
        provider = _provider()
        capabilities = provider.get_capabilities("pro")
        assert capabilities.model_name == "gemini-4-argon"
        assert capabilities.context_window == 1048576
        assert capabilities.max_output_tokens == 65536
        assert capabilities.supported_thinking_levels == ["low", "medium", "high"]

    def test_extended_reasoning_prefers_argon(self):
        """The flagship reasoning model wins EXTENDED_REASONING even though its id lacks 'pro'."""
        from tools.models import ToolModelCategory

        provider = _provider()
        preferred = provider.get_preferred_model(
            ToolModelCategory.EXTENDED_REASONING, ["gemini-3.1-pro-preview", "gemini-4-argon", "gemini-3.8-flash"]
        )
        assert preferred == "gemini-4-argon"

    def test_shut_down_flash_lite_preview_removed(self):
        """gemini-3.1-flash-lite-preview was shut down by Google on 2026-05-25."""
        assert not _provider().validate_model_name("gemini-3.1-flash-lite-preview")

    def test_extended_reasoning_ranks_by_intelligence_score(self, monkeypatch):
        """A future flagship without 'pro' in its id wins on intelligence_score, with no code edit."""
        from tools.models import ToolModelCategory

        provider = _provider()
        capability_map = dict(provider.get_all_model_capabilities())
        capability_map["gemini-5-future"] = _capabilities("gemini-5-future", intelligence_score=20)
        monkeypatch.setattr(provider, "get_all_model_capabilities", lambda: capability_map)

        preferred = provider.get_preferred_model(
            ToolModelCategory.EXTENDED_REASONING, ["gemini-4-argon", "gemini-3.1-pro-preview", "gemini-5-future"]
        )
        assert preferred == "gemini-5-future"

    def test_extended_reasoning_score_tie_still_prefers_pro(self):
        """Equal scores keep the existing 'pro' preference, newest id first."""
        from tools.models import ToolModelCategory

        preferred = _provider().get_preferred_model(
            ToolModelCategory.EXTENDED_REASONING, ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-3.8-flash"]
        )
        assert preferred == "gemini-3.1-pro-preview"
