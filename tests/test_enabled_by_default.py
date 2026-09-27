"""``enabled_by_default: false`` keeps a registry entry out of discovery (#149).

A disabled entry must not reach auto mode, the ranked summary, ``list_models``
or alias resolution, yet it must still answer to its exact canonical name and
come back into discovery when the provider's allowlist names it. Each test
runs against a two-entry OpenAI or OpenRouter config written to ``tmp_path``,
so the assertions do not depend on what conf/ happens to ship.
"""

from __future__ import annotations

import json

import pytest

import utils.model_restrictions as model_restrictions
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from tools.chat import ChatTool

ENABLED = "fixture-enabled"
DISABLED = "fixture-disabled"


def _write_config(tmp_path, entries):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": entries}))
    return path


def _entry(name, alias, *, enabled, score):
    entry = {
        "model_name": name,
        "aliases": [alias],
        "context_window": 200_000,
        "max_output_tokens": 64_000,
        "intelligence_score": score,
    }
    if not enabled:
        entry["enabled_by_default"] = False
    return entry


@pytest.fixture
def openai_fixture(tmp_path, monkeypatch):
    # The disabled entry outranks the enabled one, so a ranking or fallback
    # that ignored the flag would pick it first.
    path = _write_config(
        tmp_path,
        [
            _entry(ENABLED, "fixture-on", enabled=True, score=10),
            _entry(DISABLED, "fixture-off", enabled=False, score=20),
        ],
    )
    monkeypatch.setenv("OPENAI_MODELS_CONFIG_PATH", str(path))
    OpenAIModelProvider.reload_registry()

    registry = ModelProviderRegistry()
    registry._providers.clear()
    registry._initialized_providers.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    model_restrictions._restriction_service = None

    yield ModelProviderRegistry.get_provider(ProviderType.OPENAI)

    monkeypatch.delenv("OPENAI_MODELS_CONFIG_PATH")
    OpenAIModelProvider.reload_registry()


def _allow(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    model_restrictions._restriction_service = None


class TestDisabledEntryHidden:
    def test_excluded_from_list_models(self, openai_fixture):
        assert openai_fixture.list_models() == [ENABLED, "fixture-on"]
        assert DISABLED not in openai_fixture.list_models(respect_restrictions=False)

    def test_excluded_from_ranking(self, openai_fixture):
        assert [name for name, _ in openai_fixture.get_capabilities_by_rank()] == [ENABLED]

    def test_excluded_from_available_models(self, openai_fixture):
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        assert ENABLED in available
        assert DISABLED not in available

    def test_excluded_from_auto_mode_fallback(self, openai_fixture):
        assert ModelProviderRegistry.get_preferred_fallback_model() == ENABLED

    def test_excluded_from_ranked_summary(self, openai_fixture):
        summaries, total, _ = ChatTool()._get_ranked_model_summaries()
        assert total == 1
        assert summaries[0].startswith(f"{ENABLED} (")

    def test_alias_does_not_resolve(self, openai_fixture):
        assert openai_fixture.validate_model_name("fixture-on")
        assert not openai_fixture.validate_model_name("fixture-off")

    def test_exact_canonical_name_still_served(self, openai_fixture):
        assert openai_fixture.get_capabilities(DISABLED).model_name == DISABLED


class TestAllowlistOptsIn:
    def test_canonical_name_in_allowlist_enables_discovery(self, openai_fixture, monkeypatch):
        _allow(monkeypatch, "OPENAI_ALLOWED_MODELS", f"{ENABLED},{DISABLED}")

        assert DISABLED in openai_fixture.list_models()
        assert DISABLED in ModelProviderRegistry.get_available_models(respect_restrictions=True)
        # Now discoverable, it outranks the enabled entry.
        summaries, _, _ = ChatTool()._get_ranked_model_summaries()
        assert summaries[0].startswith(f"{DISABLED} (")

    def test_allowlist_without_it_keeps_it_hidden_and_rejected(self, openai_fixture, monkeypatch):
        _allow(monkeypatch, "OPENAI_ALLOWED_MODELS", ENABLED)

        assert DISABLED not in ModelProviderRegistry.get_available_models(respect_restrictions=True)
        assert not openai_fixture.validate_model_name(DISABLED)

    def test_alias_in_allowlist_does_not_enable(self, openai_fixture, monkeypatch):
        _allow(monkeypatch, "OPENAI_ALLOWED_MODELS", "fixture-off")

        assert ModelProviderRegistry.get_available_models(respect_restrictions=True) == {}


class TestOpenRouter:
    @pytest.fixture
    def openrouter(self, tmp_path, monkeypatch):
        path = _write_config(
            tmp_path,
            [
                _entry("vendor/on", "or-on", enabled=True, score=10),
                _entry("vendor/off", "or-off", enabled=False, score=20),
            ],
        )
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(path))
        # The registry is class-level and built once per process; rebuild it
        # from the fixture config and let monkeypatch put the shared one back.
        monkeypatch.setattr(OpenRouterProvider, "_registry", None)
        return OpenRouterProvider(api_key="test-key")

    def test_list_models_skips_disabled(self, openrouter):
        assert openrouter.list_models() == ["vendor/on"]
        assert openrouter.list_models(respect_restrictions=False) == ["vendor/on", "or-on"]

    def test_alias_does_not_resolve_but_name_does(self, openrouter):
        assert openrouter._resolve_model_name("or-off") == "or-off"
        capabilities = openrouter.get_capabilities("vendor/off")
        # Registry capabilities, not the generic fallback for unknown names.
        assert capabilities.intelligence_score == 20
        assert not getattr(capabilities, "_is_generic", False)

    def test_allowlist_enables(self, openrouter, monkeypatch):
        _allow(monkeypatch, "OPENROUTER_ALLOWED_MODELS", "vendor/off")
        assert openrouter.list_models() == ["vendor/off"]


class TestProviderRouting:
    """A disabled entry must not take a name that an enabled entry elsewhere serves.

    OpenAI outranks OpenRouter in provider priority. Here OpenAI has a disabled
    entry named ``fixture-disabled`` and OpenRouter an enabled entry with that
    name as an alias; the alias has to keep winning.
    """

    @pytest.fixture
    def openrouter_alias(self, tmp_path, monkeypatch, openai_fixture):
        path = tmp_path / "openrouter.json"
        path.write_text(json.dumps({"models": [_entry("vendor/on", DISABLED, enabled=True, score=10)]}))
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(path))
        monkeypatch.setattr(OpenRouterProvider, "_registry", None)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

    def test_enabled_alias_in_lower_priority_provider_wins(self, openrouter_alias):
        provider = ModelProviderRegistry.get_provider_for_model(DISABLED)
        assert provider.get_provider_type() == ProviderType.OPENROUTER

    def test_disabled_entry_serves_when_nothing_else_does(self, openai_fixture):
        provider = ModelProviderRegistry.get_provider_for_model(DISABLED)
        assert provider.get_provider_type() == ProviderType.OPENAI

    def test_opted_in_entry_keeps_its_priority(self, openrouter_alias, monkeypatch):
        _allow(monkeypatch, "OPENAI_ALLOWED_MODELS", DISABLED)
        provider = ModelProviderRegistry.get_provider_for_model(DISABLED)
        assert provider.get_provider_type() == ProviderType.OPENAI
