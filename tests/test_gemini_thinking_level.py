"""
Gemini 3.x gets a thinking_level; older Gemini keeps its thinking_budget.

Google deprecated the numeric thinking_budget across Gemini 3.x; the supported
control is thinking_level, whose top value is "high". Sending a budget meant even
thinking_mode="max" asked 3.x models for less reasoning than their top level.
"""

import pytest
from google.genai import types

from providers.gemini import GeminiModelProvider


def _thinking_config(model_name, thinking_mode):
    provider = GeminiModelProvider(api_key="dummy-key-for-tests")
    request = provider._build_request(
        prompt="hello",
        model_name=model_name,
        system_prompt=None,
        temperature=1.0,
        max_output_tokens=None,
        thinking_mode=thinking_mode,
    )
    return request["config"].thinking_config


class TestGemini3ThinkingLevel:
    @pytest.mark.parametrize(
        ("thinking_mode", "level"),
        [
            ("max", types.ThinkingLevel.HIGH),
            ("high", types.ThinkingLevel.HIGH),
            ("medium", types.ThinkingLevel.MEDIUM),
            ("low", types.ThinkingLevel.LOW),
            # 3.8 Flash and 3.1 Pro refuse minimal, so it rounds up.
            ("minimal", types.ThinkingLevel.LOW),
        ],
    )
    def test_gemini_3_sends_thinking_level_not_budget(self, thinking_mode, level):
        config = _thinking_config("gemini-3.8-flash", thinking_mode)
        assert config.thinking_level == level
        assert config.thinking_budget is None

    def test_alias_resolves_before_the_generation_check(self):
        config = _thinking_config("flash", "max")
        assert config.thinking_level == types.ThinkingLevel.HIGH

    def test_pro_preview_also_gets_a_level(self):
        config = _thinking_config("gemini-3.1-pro-preview", "high")
        assert config.thinking_level == types.ThinkingLevel.HIGH

    def test_unknown_mode_sends_no_thinking_config(self):
        assert _thinking_config("gemini-3.8-flash", "bogus") is None


class TestGemini25ThinkingBudget:
    @pytest.mark.parametrize(("thinking_mode", "budget"), [("max", 24_576), ("high", 16_465), ("minimal", 122)])
    def test_budget_is_a_fraction_of_the_registry_ceiling(self, thinking_mode, budget):
        config = _thinking_config("gemini-2.5-flash", thinking_mode)
        assert config.thinking_budget == budget
        assert config.thinking_level is None

    def test_unknown_mode_sends_no_thinking_config(self):
        assert _thinking_config("gemini-2.5-flash", "bogus") is None
