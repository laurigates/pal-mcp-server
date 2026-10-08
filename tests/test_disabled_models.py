"""``DISABLED_MODELS`` is a global blocklist applied after the allowlists (#150).

A blocked model must disappear from auto mode, the ranked summary and
``listmodels``, and be rejected when named explicitly. Entries are canonical
names or aliases; ``provider:model`` scopes an entry to one provider. Runs
against the shipped XAI, OpenAI and OpenRouter manifests: ``grok`` is an alias
of ``grok-4.7`` on XAI and of ``x-ai/grok-4.6`` on OpenRouter.
"""

from __future__ import annotations

import json
import logging

import pytest

import server
import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from providers.registry import PROVIDER_CLASS_BY_TYPE, ModelProviderRegistry
from providers.shared import ProviderType
from tools.chat import ChatTool
from tools.listmodels import ListModelsTool


def _set(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    model_restrictions._restriction_service = None


def _service():
    return model_restrictions.get_restriction_service()


@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
    return ModelProviderRegistry.get_provider(ProviderType.OPENROUTER)


class TestIsAllowed:
    def test_canonical_name_is_blocked(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5-nano")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5-nano")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5-mini")

    def test_alias_blocks_the_model_it_resolves_to(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok")
        assert not _service().is_allowed(ProviderType.XAI, "grok-4.7")
        assert not _service().is_allowed(ProviderType.XAI, "grok-4.7", "grok4")
        assert _service().is_allowed(ProviderType.XAI, "grok-4.6")

    def test_requesting_another_alias_of_a_blocked_model_is_blocked(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok-4.7")
        assert not _service().is_allowed(ProviderType.XAI, "grok4")

    def test_case_and_whitespace_insensitive(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", " GPT-5-Nano , ")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5-nano")

    def test_unset_blocks_nothing(self):
        assert _service().is_allowed(ProviderType.XAI, "grok-4.7")


class TestProviderScope:
    def test_scoped_entry_blocks_only_that_provider(self, monkeypatch, openrouter):
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:x-ai/grok-4.6")
        assert not _service().is_allowed(ProviderType.OPENROUTER, "x-ai/grok-4.6")
        assert _service().is_allowed(ProviderType.XAI, "grok-4.6")

    def test_scoped_alias_leaves_other_providers_resolution_alone(self, monkeypatch, openrouter):
        # ``grok`` resolves on both providers, so only the scope keeps the
        # entry off the other one.
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:grok")
        assert not _service().is_allowed(ProviderType.OPENROUTER, "x-ai/grok-4.6")
        assert _service().is_allowed(ProviderType.XAI, "grok-4.7")

        _set(monkeypatch, "DISABLED_MODELS", "xai:grok")
        assert not _service().is_allowed(ProviderType.XAI, "grok-4.7")
        assert _service().is_allowed(ProviderType.OPENROUTER, "x-ai/grok-4.6")

    def test_scoped_entry_covers_the_providers_aliases(self, monkeypatch, openrouter):
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:x-ai/grok-4.6")
        assert not openrouter.validate_model_name("grok")
        # The alias loop in OpenRouterProvider.list_models must not re-admit it.
        assert "x-ai/grok-4.6" not in openrouter.list_models()
        assert ModelProviderRegistry.get_provider(ProviderType.XAI).validate_model_name("grok-4.6")

    def test_colon_without_a_provider_prefix_is_part_of_the_name(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "llama3.2:1b")
        assert not _service().is_allowed(ProviderType.CUSTOM, "llama3.2:1b")
        assert _service().is_allowed(ProviderType.CUSTOM, "llama3.2")


class TestHiddenEverywhere:
    def test_rejected_when_named_explicitly(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok,gpt-5-nano")
        xai = ModelProviderRegistry.get_provider(ProviderType.XAI)
        openai = ModelProviderRegistry.get_provider(ProviderType.OPENAI)
        assert not xai.validate_model_name("grok")
        assert not xai.validate_model_name("grok-4.7")
        assert not openai.validate_model_name("nano")
        with pytest.raises(ValueError, match="not allowed by restriction policy"):
            xai.get_capabilities("grok-4.7")

    def test_hidden_from_available_models(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok,gpt-5-nano")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        for name in ("grok-4.7", "grok", "grok4", "gpt-5-nano", "nano"):
            assert name not in available
        assert "grok-4.6" in available

    def test_hidden_from_auto_mode_fallback(self, monkeypatch):
        registry = ModelProviderRegistry()
        registry._providers.clear()
        registry._initialized_providers.clear()
        ModelProviderRegistry.register_provider(ProviderType.XAI, PROVIDER_CLASS_BY_TYPE[ProviderType.XAI])
        _set(monkeypatch, "DISABLED_MODELS", "")
        unblocked = ModelProviderRegistry.get_preferred_fallback_model()

        _set(monkeypatch, "DISABLED_MODELS", unblocked)
        fallback = ModelProviderRegistry.get_preferred_fallback_model()
        assert fallback != unblocked
        assert ModelProviderRegistry.get_provider(ProviderType.XAI).validate_model_name(fallback)

    def test_hidden_from_ranked_summary(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok")
        summaries, _, _ = ChatTool()._get_ranked_model_summaries(limit=100)
        assert not any(s.startswith("grok-4.7 ") for s in summaries)
        assert any(s.startswith("grok-4.6 ") for s in summaries)

    @pytest.mark.asyncio
    async def test_hidden_from_listmodels(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "grok,gpt-5-nano")
        content = json.loads((await ListModelsTool().execute({}))[0].text)["content"]
        assert "`grok-4.7`" not in content
        assert "`grok` →" not in content
        assert "`gpt-5-nano`" not in content
        assert "`grok-4.6`" in content


class TestAllowlistInteraction:
    def test_blocklist_applies_after_allowlist(self, monkeypatch):
        _set(monkeypatch, "XAI_ALLOWED_MODELS", "grok-4.7,grok-4.6")
        _set(monkeypatch, "DISABLED_MODELS", "grok")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        xai_models = sorted(name for name, ptype in available.items() if ptype == ProviderType.XAI)
        assert xai_models == ["grok-4.6"]
        assert not _service().is_allowed(ProviderType.XAI, "grok-4.7")
        assert _service().is_allowed(ProviderType.XAI, "grok-4.6")

    def test_blocking_the_whole_allowlist_leaves_nothing(self, monkeypatch):
        _set(monkeypatch, "XAI_ALLOWED_MODELS", "grok")
        _set(monkeypatch, "DISABLED_MODELS", "grok-4.7")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        assert ProviderType.XAI not in available.values()

    def test_allowlist_still_applies_to_unblocked_models(self, monkeypatch):
        _set(monkeypatch, "XAI_ALLOWED_MODELS", "grok-4.6")
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5-nano")
        assert not _service().is_allowed(ProviderType.XAI, "grok-4.7")
        assert _service().is_allowed(ProviderType.XAI, "grok-4.6")


class TestStartupValidation:
    def _providers(self):
        return {
            ptype: ModelProviderRegistry.get_provider(ptype)
            for ptype in (ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.XAI)
        }

    def test_unknown_name_warns(self, monkeypatch, caplog):
        _set(monkeypatch, "DISABLED_MODELS", "grok,no-such-model")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            _service().validate_against_known_models(self._providers())
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no-such-model" in w and "DISABLED_MODELS" in w for w in warnings)
        assert not any("'grok'" in w for w in warnings)

    def test_known_blocked_names_do_not_warn(self, monkeypatch, caplog):
        # validate_model_name rejects a blocked model, so validation must not
        # use it to decide whether the name is known.
        _set(monkeypatch, "DISABLED_MODELS", "grok,gpt-5-nano,xai:grok-4.6")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            _service().validate_against_known_models(self._providers())
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_configure_providers_warns_and_starts(self, monkeypatch, caplog):
        _set(monkeypatch, "DISABLED_MODELS", "no-such-model")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            server.configure_providers()
        assert any("no-such-model" in r.getMessage() for r in caplog.records)
