"""``PREFERRED_MODELS`` is an ordered list auto mode prefers over the ranking (#151).

The auto-mode fallback returns the first available, allowed entry regardless
of tool category, and the ranked summary in tool descriptions lists the
preferred models first, in the user's order. Entries resolve through aliases
and respect ``*_ALLOWED_MODELS`` and ``DISABLED_MODELS``. Runs against the
shipped Google, OpenAI and XAI manifests registered by ``conftest``.
"""

from __future__ import annotations

import json
import logging

import pytest

import server
import utils.model_restrictions as model_restrictions
from providers.openai import OpenAIModelProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from tools.chat import ChatTool
from tools.models import ToolModelCategory


def _set(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    model_restrictions._restriction_service = None


def _summary_names(limit=5):
    summaries, total, _ = ChatTool()._get_ranked_model_summaries(limit=limit)
    return [s.split(" (", 1)[0] for s in summaries], total


class TestFallback:
    @pytest.mark.parametrize("category", [None, *ToolModelCategory])
    def test_first_preferred_model_wins_in_every_category(self, monkeypatch, category):
        _set(monkeypatch, "PREFERRED_MODELS", "grok-4.3,gpt-5-mini")
        assert ModelProviderRegistry.get_preferred_fallback_model(category) == "grok-4.3"

    def test_alias_resolves_to_canonical_name(self, monkeypatch):
        _set(monkeypatch, "PREFERRED_MODELS", "nano")
        assert ModelProviderRegistry.get_preferred_fallback_model() == "gpt-5-nano"

    def test_unknown_and_unconfigured_entries_are_skipped(self, monkeypatch):
        # x-ai/grok-4.6 is an OpenRouter model; OpenRouter is not configured.
        _set(monkeypatch, "PREFERRED_MODELS", "no-such-model,x-ai/grok-4.6,gpt-5-mini")
        assert ModelProviderRegistry.get_preferred_fallback_model() == "gpt-5-mini"

    def test_disabled_models_entry_is_skipped(self, monkeypatch):
        _set(monkeypatch, "PREFERRED_MODELS", "grok-4.3,gpt-5-mini")
        _set(monkeypatch, "DISABLED_MODELS", "grok-4.3")
        assert ModelProviderRegistry.get_preferred_fallback_model() == "gpt-5-mini"

    def test_entry_outside_the_allowlist_is_skipped(self, monkeypatch):
        _set(monkeypatch, "PREFERRED_MODELS", "grok-4.3,gpt-5-mini")
        _set(monkeypatch, "XAI_ALLOWED_MODELS", "grok-4.6")
        assert ModelProviderRegistry.get_preferred_fallback_model() == "gpt-5-mini"

    def test_no_available_entry_falls_back_to_provider_preference(self, monkeypatch):
        default = ModelProviderRegistry.get_preferred_fallback_model()
        _set(monkeypatch, "PREFERRED_MODELS", "no-such-model")
        assert ModelProviderRegistry.get_preferred_fallback_model() == default


class TestRankedSummary:
    def test_preferred_models_come_first_in_user_order(self, monkeypatch):
        default_names, default_total = _summary_names(limit=100)
        assert default_names.index("gpt-5-nano") > 5 and default_names.index("grok-4.3") > 5

        _set(monkeypatch, "PREFERRED_MODELS", "grok-4.3,no-such-model,nano")
        names, total = _summary_names(limit=100)

        assert names[:2] == ["grok-4.3", "gpt-5-nano"]
        # The rest keep their score order.
        assert names[2:] == [n for n in default_names if n not in ("grok-4.3", "gpt-5-nano")]
        assert total == default_total

    def test_preferred_models_fill_the_top_slots(self, monkeypatch):
        _set(monkeypatch, "PREFERRED_MODELS", "gpt-5-nano")
        names, _ = _summary_names()
        assert names[0] == "gpt-5-nano"
        assert len(names) == 5

    def test_blocked_preferred_model_is_not_listed(self, monkeypatch):
        _set(monkeypatch, "PREFERRED_MODELS", "gpt-5-nano")
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5-nano")
        names, _ = _summary_names(limit=100)
        assert "gpt-5-nano" not in names


class TestDisabledByDefaultOptIn:
    @pytest.fixture
    def openai_fixture(self, tmp_path, monkeypatch):
        path = tmp_path / "models.json"
        entries = [
            {
                "model_name": "fixture-on",
                "context_window": 200_000,
                "max_output_tokens": 64_000,
                "intelligence_score": 10,
            },
            {
                "model_name": "fixture-off",
                "aliases": ["fixture-off-alias"],
                "context_window": 200_000,
                "max_output_tokens": 64_000,
                "intelligence_score": 5,
                "enabled_by_default": False,
            },
        ]
        path.write_text(json.dumps({"models": entries}))
        monkeypatch.setenv("OPENAI_MODELS_CONFIG_PATH", str(path))
        OpenAIModelProvider.reload_registry()
        registry = ModelProviderRegistry()
        registry._providers.clear()
        registry._initialized_providers.clear()
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
        yield
        monkeypatch.delenv("OPENAI_MODELS_CONFIG_PATH")
        OpenAIModelProvider.reload_registry()

    def test_preferred_canonical_name_opts_it_in(self, monkeypatch, openai_fixture):
        _set(monkeypatch, "PREFERRED_MODELS", "fixture-off")
        assert "fixture-off" in ModelProviderRegistry.get_available_models(respect_restrictions=True)
        assert ModelProviderRegistry.get_preferred_fallback_model() == "fixture-off"
        assert _summary_names()[0][0] == "fixture-off"

    def test_unset_keeps_it_hidden(self, openai_fixture):
        assert "fixture-off" not in ModelProviderRegistry.get_available_models(respect_restrictions=True)


class TestStartupWarning:
    def test_unavailable_entries_warn(self, monkeypatch, caplog):
        _set(monkeypatch, "PREFERRED_MODELS", "no-such-model,grok-4.3")
        with caplog.at_level(logging.WARNING):
            server.configure_providers()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no-such-model" in w and "PREFERRED_MODELS" in w for w in warnings)
        assert not any("'grok-4.3'" in w for w in warnings)
