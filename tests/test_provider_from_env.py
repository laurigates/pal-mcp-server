"""Tests for the per-provider ``from_env()`` classmethods.

These methods encapsulate the env-var lookup, placeholder rejection, and
optional supplementary configuration that used to live inline in
``server.configure_providers``. The goal of the refactor (Issue #7) was to
collapse that monolith into a registry-driven loop, so the same env-var
contract is now exercised against the provider classes directly.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from providers.azure_openai import AzureOpenAIProvider
from providers.custom import CustomProvider
from providers.dial import DIALModelProvider
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import REGISTERED_PROVIDER_CLASSES
from providers.shared import ProviderType
from providers.xai import XAIModelProvider

pytestmark = pytest.mark.no_mock_provider


# --------------------------------------------------------------------------
# Registry list
# --------------------------------------------------------------------------


def test_registered_provider_classes_priority_order():
    """The registered class list must follow native -> custom -> catch-all order."""

    classes = list(REGISTERED_PROVIDER_CLASSES)
    assert classes == [
        GeminiModelProvider,
        OpenAIModelProvider,
        AzureOpenAIProvider,
        XAIModelProvider,
        DIALModelProvider,
        CustomProvider,
        OpenRouterProvider,
    ]


def test_every_registered_class_exposes_from_env():
    """Every registry entry must implement the env-var bridge."""

    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        assert hasattr(provider_cls, "from_env"), provider_cls
        assert callable(provider_cls.from_env), provider_cls


# --------------------------------------------------------------------------
# Single-key providers - present
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls, env_var, expected_type",
    [
        (GeminiModelProvider, "GEMINI_API_KEY", ProviderType.GOOGLE),
        (OpenAIModelProvider, "OPENAI_API_KEY", ProviderType.OPENAI),
        (XAIModelProvider, "XAI_API_KEY", ProviderType.XAI),
        (DIALModelProvider, "DIAL_API_KEY", ProviderType.DIAL),
        (OpenRouterProvider, "OPENROUTER_API_KEY", ProviderType.OPENROUTER),
    ],
)
def test_from_env_returns_instance_when_key_present(provider_cls, env_var, expected_type):
    """A non-placeholder key produces a provider whose declared type matches."""

    with patch.dict(os.environ, {env_var: "real-test-key"}, clear=False):
        instance = provider_cls.from_env()
    assert instance is not None
    assert instance.get_provider_type() is expected_type


# --------------------------------------------------------------------------
# Single-key providers - missing / placeholder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls, env_var, placeholder",
    [
        (GeminiModelProvider, "GEMINI_API_KEY", "your_gemini_api_key_here"),
        (OpenAIModelProvider, "OPENAI_API_KEY", "your_openai_api_key_here"),
        (XAIModelProvider, "XAI_API_KEY", "your_xai_api_key_here"),
        (DIALModelProvider, "DIAL_API_KEY", "your_dial_api_key_here"),
        (OpenRouterProvider, "OPENROUTER_API_KEY", "your_openrouter_api_key_here"),
    ],
)
def test_from_env_returns_none_for_missing_or_placeholder(provider_cls, env_var, placeholder):
    """Both an unset key and the documented placeholder string must be rejected."""

    cleared_env = {k: v for k, v in os.environ.items() if k != env_var}
    with patch.dict(os.environ, cleared_env, clear=True):
        assert provider_cls.from_env() is None
    with patch.dict(os.environ, {env_var: ""}, clear=False):
        assert provider_cls.from_env() is None
    with patch.dict(os.environ, {env_var: placeholder}, clear=False):
        assert provider_cls.from_env() is None


# --------------------------------------------------------------------------
# Azure - dual-requirement provider (key + endpoint + non-empty registry)
# --------------------------------------------------------------------------


def test_azure_from_env_missing_endpoint_returns_none():
    """Azure must have BOTH a key AND an endpoint configured."""

    env = {
        "AZURE_OPENAI_API_KEY": "real-test-key",
    }
    # Strip endpoint from the parent environment so this is a clean negative.
    cleared = {k: v for k, v in os.environ.items() if k != "AZURE_OPENAI_ENDPOINT"}
    cleared.update(env)
    with patch.dict(os.environ, cleared, clear=True):
        assert AzureOpenAIProvider.from_env() is None


def test_azure_from_env_missing_key_returns_none():
    """Azure must reject configurations that have an endpoint but no key."""

    env = {"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com"}
    cleared = {k: v for k, v in os.environ.items() if k != "AZURE_OPENAI_API_KEY"}
    cleared.update(env)
    with patch.dict(os.environ, cleared, clear=True):
        assert AzureOpenAIProvider.from_env() is None


def test_azure_from_env_placeholder_key_returns_none():
    """The documented placeholder key must be treated as 'not configured'."""

    env = {
        "AZURE_OPENAI_API_KEY": "your_azure_openai_key_here",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    }
    with patch.dict(os.environ, env, clear=False):
        assert AzureOpenAIProvider.from_env() is None


def test_azure_from_env_empty_model_registry_returns_none(monkeypatch):
    """Azure refuses to register itself when no deployments are configured."""

    env = {
        "AZURE_OPENAI_API_KEY": "real-test-key",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
    }

    class _EmptyRegistry:
        def list_models(self):
            return []

    monkeypatch.setattr(
        "providers.azure_openai.AzureModelRegistry",
        lambda *args, **kwargs: _EmptyRegistry(),
    )
    with patch.dict(os.environ, env, clear=False):
        assert AzureOpenAIProvider.from_env() is None


# --------------------------------------------------------------------------
# Custom provider - URL is the gate, key is optional
# --------------------------------------------------------------------------


def test_custom_from_env_missing_url_returns_none():
    """No CUSTOM_API_URL means 'don't register the custom provider'."""

    cleared = {k: v for k, v in os.environ.items() if k != "CUSTOM_API_URL"}
    with patch.dict(os.environ, cleared, clear=True):
        assert CustomProvider.from_env() is None


def test_custom_from_env_with_url_returns_instance():
    """A URL is sufficient even without a key (Ollama case)."""

    env = {"CUSTOM_API_URL": "http://localhost:11434/v1"}
    # Drop CUSTOM_API_KEY so this is the unauthenticated path.
    cleared = {k: v for k, v in os.environ.items() if k not in {"CUSTOM_API_KEY"}}
    cleared.update(env)
    with patch.dict(os.environ, cleared, clear=True):
        instance = CustomProvider.from_env()
    assert instance is not None
    assert instance.get_provider_type() is ProviderType.CUSTOM


def test_custom_from_env_with_url_and_key_returns_instance():
    """A URL with an explicit key (vLLM/LM Studio) constructs the provider."""

    env = {
        "CUSTOM_API_URL": "http://localhost:11434/v1",
        "CUSTOM_API_KEY": "real-key",
    }
    with patch.dict(os.environ, env, clear=False):
        instance = CustomProvider.from_env()
    assert instance is not None
    assert instance.get_provider_type() is ProviderType.CUSTOM
