"""``DISABLED_MODELS`` is a global blocklist applied after the allowlists (#150).

A blocked model must disappear from auto mode, the ranked summary and
``listmodels``, and be rejected when named explicitly. Entries are canonical
names or aliases; ``provider:model`` scopes an entry to one provider. Runs
against the shipped OpenAI and OpenRouter manifests: ``codex`` is an alias of
``gpt-5.3-codex`` on OpenAI and of ``openai/gpt-5.3-codex`` on OpenRouter.
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
        _set(monkeypatch, "DISABLED_MODELS", "codex")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex", "codex-5.3")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.2")

    def test_requesting_another_alias_of_a_blocked_model_is_blocked(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5.3-codex")
        assert not _service().is_allowed(ProviderType.OPENAI, "codex-5.3")

    def test_case_and_whitespace_insensitive(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", " GPT-5-Nano , ")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5-nano")

    def test_unset_blocks_nothing(self):
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")


class TestProviderScope:
    def test_scoped_entry_blocks_only_that_provider(self, monkeypatch, openrouter):
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:openai/gpt-5.3-codex")
        assert not _service().is_allowed(ProviderType.OPENROUTER, "openai/gpt-5.3-codex")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.2")

    def test_scoped_alias_leaves_other_providers_resolution_alone(self, monkeypatch, openrouter):
        # ``codex`` resolves on both providers, so only the scope keeps the
        # entry off the other one.
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:codex")
        assert not _service().is_allowed(ProviderType.OPENROUTER, "openai/gpt-5.3-codex")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")

        _set(monkeypatch, "DISABLED_MODELS", "openai:codex")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")
        assert _service().is_allowed(ProviderType.OPENROUTER, "openai/gpt-5.3-codex")

    def test_scoped_entry_covers_the_providers_aliases(self, monkeypatch, openrouter):
        _set(monkeypatch, "DISABLED_MODELS", "openrouter:openai/gpt-5.3-codex")
        assert not openrouter.validate_model_name("codex")
        # The alias loop in OpenRouterProvider.list_models must not re-admit it.
        assert "openai/gpt-5.3-codex" not in openrouter.list_models()
        assert ModelProviderRegistry.get_provider(ProviderType.OPENAI).validate_model_name("gpt-5.2")

    def test_colon_without_a_provider_prefix_is_part_of_the_name(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "llama3.2:1b")
        assert not _service().is_allowed(ProviderType.CUSTOM, "llama3.2:1b")
        assert _service().is_allowed(ProviderType.CUSTOM, "llama3.2")


class TestHiddenEverywhere:
    def test_rejected_when_named_explicitly(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "codex,gpt-5-nano")
        openai_provider = ModelProviderRegistry.get_provider(ProviderType.OPENAI)
        openai = ModelProviderRegistry.get_provider(ProviderType.OPENAI)
        assert not openai_provider.validate_model_name("codex")
        assert not openai_provider.validate_model_name("gpt-5.3-codex")
        assert not openai.validate_model_name("nano")
        with pytest.raises(ValueError, match="not allowed by restriction policy"):
            openai_provider.get_capabilities("gpt-5.3-codex")

    def test_hidden_from_available_models(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "codex,gpt-5-nano")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        for name in ("gpt-5.3-codex", "codex", "codex-5.3", "gpt-5-nano", "nano"):
            assert name not in available
        assert "gpt-5.2" in available

    def test_hidden_from_auto_mode_fallback(self, monkeypatch):
        registry = ModelProviderRegistry()
        registry._providers.clear()
        registry._initialized_providers.clear()
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, PROVIDER_CLASS_BY_TYPE[ProviderType.OPENAI])
        _set(monkeypatch, "DISABLED_MODELS", "")
        unblocked = ModelProviderRegistry.get_preferred_fallback_model()

        _set(monkeypatch, "DISABLED_MODELS", unblocked)
        fallback = ModelProviderRegistry.get_preferred_fallback_model()
        assert fallback != unblocked
        assert ModelProviderRegistry.get_provider(ProviderType.OPENAI).validate_model_name(fallback)

    def test_hidden_from_ranked_summary(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "codex")
        summaries, _, _ = ChatTool()._get_ranked_model_summaries(limit=100)
        assert not any(s.startswith("gpt-5.3-codex ") for s in summaries)
        assert any(s.startswith("gpt-5.2 ") for s in summaries)

    @pytest.mark.asyncio
    async def test_hidden_from_listmodels(self, monkeypatch):
        _set(monkeypatch, "DISABLED_MODELS", "codex,gpt-5-nano")
        content = json.loads((await ListModelsTool().execute({}))[0].text)["content"]
        assert "`gpt-5.3-codex`" not in content
        assert "`codex` →" not in content
        assert "`gpt-5-nano`" not in content
        assert "`gpt-5.2`" in content


class TestAllowlistInteraction:
    def test_blocklist_applies_after_allowlist(self, monkeypatch):
        _set(monkeypatch, "OPENAI_ALLOWED_MODELS", "gpt-5.3-codex,gpt-5.2")
        _set(monkeypatch, "DISABLED_MODELS", "codex")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        openai_models = sorted(name for name, ptype in available.items() if ptype == ProviderType.OPENAI)
        assert openai_models == ["gpt-5.2"]
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.2")

    def test_blocking_the_whole_allowlist_leaves_nothing(self, monkeypatch):
        _set(monkeypatch, "OPENAI_ALLOWED_MODELS", "codex")
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5.3-codex")
        available = ModelProviderRegistry.get_available_models(respect_restrictions=True)
        assert ProviderType.OPENAI not in available.values()

    def test_allowlist_still_applies_to_unblocked_models(self, monkeypatch):
        _set(monkeypatch, "OPENAI_ALLOWED_MODELS", "gpt-5.2")
        _set(monkeypatch, "DISABLED_MODELS", "gpt-5-nano")
        assert not _service().is_allowed(ProviderType.OPENAI, "gpt-5.3-codex")
        assert _service().is_allowed(ProviderType.OPENAI, "gpt-5.2")


class TestStartupValidation:
    def _providers(self):
        return {
            ptype: ModelProviderRegistry.get_provider(ptype) for ptype in (ProviderType.GOOGLE, ProviderType.OPENAI)
        }

    def test_unknown_name_warns(self, monkeypatch, caplog):
        _set(monkeypatch, "DISABLED_MODELS", "codex,no-such-model")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            _service().validate_against_known_models(self._providers())
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("no-such-model" in w and "DISABLED_MODELS" in w for w in warnings)
        assert not any("'codex'" in w for w in warnings)

    def test_known_blocked_names_do_not_warn(self, monkeypatch, caplog):
        # validate_model_name rejects a blocked model, so validation must not
        # use it to decide whether the name is known.
        _set(monkeypatch, "DISABLED_MODELS", "codex,gpt-5-nano,openai:gpt-5.2")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            _service().validate_against_known_models(self._providers())
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_configure_providers_warns_and_starts(self, monkeypatch, caplog):
        _set(monkeypatch, "DISABLED_MODELS", "no-such-model")
        with caplog.at_level(logging.WARNING, logger="utils.model_restrictions"):
            server.configure_providers()
        assert any("no-such-model" in r.getMessage() for r in caplog.records)
