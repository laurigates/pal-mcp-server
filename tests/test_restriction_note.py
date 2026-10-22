"""Tests for BaseTool._get_restriction_note (#188)."""

import pytest

from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from tools.chat import ChatTool
from utils.model_restrictions import ModelRestrictionService

_POLICY_ENV = ("DISABLED_MODELS",) + tuple(
    env_var for provider_type in ProviderType for env_var in ModelRestrictionService.env_vars_for(provider_type)
)


@pytest.fixture
def clean_policy_env(monkeypatch):
    for env_var in _POLICY_ENV:
        monkeypatch.delenv(env_var, raising=False)
    return monkeypatch


def _note():
    return ChatTool()._get_restriction_note()


def _label(provider_type: ProviderType) -> str:
    return ModelProviderRegistry.provider_class_for(provider_type).display_label()


def test_no_policy_means_no_note(clean_policy_env):
    assert _note() is None


def test_note_covers_opencode_go_allowlist(clean_policy_env):
    clean_policy_env.setenv("OPENCODE_GO_ALLOWED_MODELS", "kimi-k2, glm-4.6")

    note = _note()

    assert note is not None
    assert f"{_label(ProviderType.OPENCODE_GO)}: glm-4.6, kimi-k2" in note


def test_note_covers_azure_documented_variable(clean_policy_env):
    clean_policy_env.setenv("AZURE_OPENAI_ALLOWED_MODELS", "gpt-4o")

    note = _note()

    assert note is not None
    assert f"{_label(ProviderType.AZURE)}: gpt-4o" in note


def test_note_covers_custom_allowlist(clean_policy_env):
    clean_policy_env.setenv("CUSTOM_ALLOWED_MODELS", "local-llama")

    note = _note()

    assert note is not None
    assert f"{_label(ProviderType.CUSTOM)}: local-llama" in note


def test_note_lists_disabled_models(clean_policy_env):
    clean_policy_env.setenv("DISABLED_MODELS", "grok, openrouter:x-ai/grok-4.6")

    note = _note()

    assert note is not None
    assert "Policy allows only" not in note
    assert "disabled: grok, openrouter:x-ai/grok-4.6" in note.lower()


def test_note_combines_allowlists_and_blocklist(clean_policy_env):
    clean_policy_env.setenv("OPENAI_ALLOWED_MODELS", "o3-mini")
    clean_policy_env.setenv("DISABLED_MODELS", "gpt-5-nano")

    note = _note()

    assert note is not None
    assert f"Policy allows only → {_label(ProviderType.OPENAI)}: o3-mini" in note
    assert "gpt-5-nano" in note


def test_whitespace_only_values_produce_no_note(clean_policy_env):
    clean_policy_env.setenv("OPENAI_ALLOWED_MODELS", " , ")
    clean_policy_env.setenv("DISABLED_MODELS", " ")

    assert _note() is None


def test_note_falls_through_blank_azure_variable_to_fallback(clean_policy_env):
    clean_policy_env.setenv("AZURE_OPENAI_ALLOWED_MODELS", " ")
    clean_policy_env.setenv("AZURE_ALLOWED_MODELS", "gpt-4o")

    note = _note()

    assert note is not None
    assert f"{_label(ProviderType.AZURE)}: gpt-4o" in note
