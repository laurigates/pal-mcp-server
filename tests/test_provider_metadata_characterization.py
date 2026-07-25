"""Characterization tests for provider metadata consumers (issue #60).

Written to pin the behaviour of every hand-maintained provider roster
BEFORE it is replaced by a derived one. Expectations here are independent
restatements of what the code did on ``main``; the refactor must keep them
green except where a delta is explicitly annotated.
"""

from __future__ import annotations

import json
import os
import re
from unittest.mock import patch

import pytest

from providers.azure_openai import AzureOpenAIProvider
from providers.custom import CustomProvider
from providers.dial import DIALModelProvider
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.opencode_go import OpenCodeGoProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import REGISTERED_PROVIDER_CLASSES, ModelProviderRegistry
from providers.shared import ProviderType
from providers.xai import XAIModelProvider
from tools.listmodels import ListModelsTool

pytestmark = pytest.mark.no_mock_provider


EXPECTED_CLASS_ORDER = [
    GeminiModelProvider,
    OpenAIModelProvider,
    AzureOpenAIProvider,
    XAIModelProvider,
    DIALModelProvider,
    CustomProvider,
    OpenCodeGoProvider,
    OpenRouterProvider,
]

EXPECTED_PRIORITY_ORDER = [
    ProviderType.GOOGLE,
    ProviderType.OPENAI,
    ProviderType.AZURE,
    ProviderType.XAI,
    ProviderType.DIAL,
    ProviderType.CUSTOM,
    ProviderType.OPENCODE_GO,
    ProviderType.OPENROUTER,
]

#: Copied verbatim from the ``key_mapping`` local that used to live in
#: ``ModelProviderRegistry._get_api_key_for_provider``.
EXPECTED_API_KEY_ENV = {
    ProviderType.GOOGLE: "GEMINI_API_KEY",
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.AZURE: "AZURE_OPENAI_API_KEY",
    ProviderType.XAI: "XAI_API_KEY",
    ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
    ProviderType.CUSTOM: "CUSTOM_API_KEY",
    ProviderType.DIAL: "DIAL_API_KEY",
    ProviderType.OPENCODE_GO: "OPENCODE_API_KEY",
}

#: The literal each provider's ``from_env`` used to hardcode inline.
EXPECTED_PLACEHOLDERS = {
    ProviderType.GOOGLE: "your_gemini_api_key_here",
    ProviderType.OPENAI: "your_openai_api_key_here",
    ProviderType.AZURE: "your_azure_openai_key_here",
    ProviderType.XAI: "your_xai_api_key_here",
    ProviderType.DIAL: "your_dial_api_key_here",
    ProviderType.OPENCODE_GO: "your_opencode_api_key_here",
    ProviderType.OPENROUTER: "your_openrouter_api_key_here",
    ProviderType.CUSTOM: None,  # URL-gated, no placeholder
}

#: ``## `` headers emitted by listmodels, in order, including "Summary".
EXPECTED_SECTION_ORDER = [
    "Google Gemini",
    "OpenAI",
    "Azure OpenAI",
    "X.AI (Grok)",
    "AI DIAL",
    "OpenCode Go",
    "OpenRouter",
    "Custom/Local API",
    "Summary",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered_provider_classes_order_is_stable():
    assert list(REGISTERED_PROVIDER_CLASSES) == EXPECTED_CLASS_ORDER


def test_priority_order_matches_registered_class_order():
    """The invariant that makes deriving one from the other a no-op."""
    assert list(ModelProviderRegistry.PROVIDER_PRIORITY_ORDER) == EXPECTED_PRIORITY_ORDER


@pytest.mark.parametrize(
    "provider_type,env_var",
    sorted(EXPECTED_API_KEY_ENV.items(), key=lambda item: item[0].value),
)
def test_api_key_env_var_per_provider(provider_type, env_var):
    """_get_api_key_for_provider reads exactly this var, raw (no placeholder
    rejection - the CUSTOM empty-key path depends on that)."""
    with patch.dict(os.environ, {env_var: f"sentinel-{provider_type.value}"}, clear=True):
        assert ModelProviderRegistry._get_api_key_for_provider(provider_type) == f"sentinel-{provider_type.value}"


@pytest.mark.parametrize(
    "provider_cls",
    EXPECTED_CLASS_ORDER,
    ids=[cls.__name__ for cls in EXPECTED_CLASS_ORDER],
)
def test_from_env_rejects_the_documented_placeholder(provider_cls):
    provider_type = provider_cls.provider_type()
    placeholder = EXPECTED_PLACEHOLDERS[provider_type]
    if placeholder is None:
        pytest.skip("Custom provider is URL-gated and has no placeholder")
    with patch.dict(os.environ, {EXPECTED_API_KEY_ENV[provider_type]: placeholder}, clear=True):
        assert provider_cls.from_env() is None


def test_declared_metadata_matches_the_deleted_tables():
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        provider_type = provider_cls.provider_type()
        assert provider_cls.API_KEY_ENV == EXPECTED_API_KEY_ENV[provider_type], provider_cls
        assert provider_cls.API_KEY_PLACEHOLDER == EXPECTED_PLACEHOLDERS[provider_type], provider_cls


def test_friendly_names_snapshot():
    """DELTA (intentional): OpenAIModelProvider used to inherit
    "OpenAI Compatible" from OpenAICompatibleProvider, so every OpenAI API
    error read "OpenAI Compatible API error for model ...". It now declares
    "OpenAI"."""
    assert [cls.FRIENDLY_NAME for cls in REGISTERED_PROVIDER_CLASSES] == [
        "Gemini",
        "OpenAI",  # was the inherited "OpenAI Compatible"
        "Azure OpenAI",
        "X.AI",
        "DIAL",
        "Custom API",
        "OpenCode Go",
        "OpenRouter",
    ]


def test_allowed_models_env_var_derivation():
    """7 of 8 derive from ProviderType; Azure declares an override chain."""
    assert GeminiModelProvider.allowed_models_env_vars() == ("GOOGLE_ALLOWED_MODELS",)
    assert OpenCodeGoProvider.allowed_models_env_vars() == ("OPENCODE_GO_ALLOWED_MODELS",)
    assert CustomProvider.allowed_models_env_vars() == ("CUSTOM_ALLOWED_MODELS",)
    assert AzureOpenAIProvider.allowed_models_env_vars() == (
        "AZURE_OPENAI_ALLOWED_MODELS",
        "AZURE_ALLOWED_MODELS",
    )


def test_gating_env_vars_capture_the_awkward_providers():
    """Activation is not always "has API key"; this is the metadata that lets
    consumers stop special-casing Custom and Azure by name."""
    assert CustomProvider.gating_env_vars() == ("CUSTOM_API_URL",)
    assert AzureOpenAIProvider.gating_env_vars() == ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
    assert OpenCodeGoProvider.gating_env_vars() == ("OPENCODE_API_KEY",)


# ---------------------------------------------------------------------------
# listmodels roster
# ---------------------------------------------------------------------------


async def _listmodels_content() -> str:
    result = await ListModelsTool().execute({})
    return json.loads(result[0].text)["content"]


@pytest.mark.asyncio
async def test_listmodels_section_order_is_exact():
    """tests/test_listmodels.py asserts substrings, which cannot catch a
    section being reordered, duplicated or dropped."""
    with patch.dict(os.environ, {"DEFAULT_MODEL": "auto"}, clear=True):
        content = await _listmodels_content()
    assert re.findall(r"^## (.+?)(?: [✅❌])?$", content, flags=re.MULTILINE) == EXPECTED_SECTION_ORDER


@pytest.mark.asyncio
async def test_listmodels_not_configured_lines_name_the_right_env_vars():
    """DELTA (intentional): Azure's line gains AZURE_OPENAI_ENDPOINT, which is
    genuinely required for the provider to activate and was never mentioned."""
    with patch.dict(os.environ, {"DEFAULT_MODEL": "auto"}, clear=True):
        content = await _listmodels_content()

    assert re.findall(r"^\*\*Status\*\*: Not configured \(set (.+?)\)$", content, flags=re.MULTILINE) == [
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT",  # was "AZURE_OPENAI_API_KEY"
        "XAI_API_KEY",
        "DIAL_API_KEY",
        "OPENCODE_API_KEY",
        "OPENROUTER_API_KEY",
        "CUSTOM_API_URL",
    ]


@pytest.mark.asyncio
async def test_listmodels_configured_count_matches_the_checkmarks():
    """The count and the check marks are computed by two separate passes; this
    invariant is what makes collapsing them onto one derived list safe."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}, clear=True):
        content = await _listmodels_content()

    checkmarks = len(re.findall(r"^## .+ ✅$", content, flags=re.MULTILINE))
    declared = int(re.search(r"\*\*Configured Providers\*\*: (\d+)", content).group(1))
    assert checkmarks == declared == 1


# ---------------------------------------------------------------------------
# server bootstrap
# ---------------------------------------------------------------------------


def test_no_provider_error_names_every_provider(monkeypatch):
    """DELTA (intentional): the message used to omit AZURE_OPENAI_API_KEY
    entirely, so a user with only an Azure subscription was told Azure was
    not an option. It is now generated from the roster."""
    import server

    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        for var in provider_cls.credential_env_vars():
            monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError) as exc_info:
        server.configure_providers()

    message = str(exc_info.value)
    assert message.startswith("At least one API configuration is required. Please set either:")
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        for var in provider_cls.gating_env_vars():
            assert var in message, f"{var} missing from bootstrap help text"
    # Phrases preserved verbatim from the hand-written text.
    assert "for X.AI GROK models" in message
    assert "for OpenRouter (multiple models)" in message
    assert "for local models (Ollama, vLLM, etc.)" in message


# ---------------------------------------------------------------------------
# Issue #66 - allow-lists must reach listing time for every provider
# ---------------------------------------------------------------------------


def test_opencode_go_allowlist_is_honoured_at_listing_time(monkeypatch):
    """OPENCODE_GO_ALLOWED_MODELS must filter the roster, not just generate()."""
    import utils.model_restrictions as model_restrictions

    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_GO_ALLOWED_MODELS", "glm-5.2")
    model_restrictions._restriction_service = None

    service = model_restrictions.get_restriction_service()
    assert service.has_restrictions(ProviderType.OPENCODE_GO) is True
    assert service.is_allowed(ProviderType.OPENCODE_GO, "glm-5.2") is True
    assert service.is_allowed(ProviderType.OPENCODE_GO, "deepseek-v4-pro") is False

    ModelProviderRegistry.register_provider(ProviderType.OPENCODE_GO, OpenCodeGoProvider)
    assert ModelProviderRegistry.get_available_model_names(ProviderType.OPENCODE_GO) == ["glm-5.2"]


# ---------------------------------------------------------------------------
# Issue #67
# ---------------------------------------------------------------------------


def test_kimi_k2_7_code_declares_no_temperature_support():
    """Issue #67: the model 400s when a temperature is sent at all."""
    from providers.registries.opencode_go import OpenCodeGoModelRegistry

    registry = OpenCodeGoModelRegistry()
    for name in ("kimi-k2.7-code", "kimi", "kimi-code"):
        capabilities = registry.resolve(name)
        assert capabilities is not None, name
        assert capabilities.supports_temperature is False, name
        # None => the caller must omit the parameter entirely.
        assert capabilities.get_effective_temperature(0.7) is None, name


def test_temperature_supporting_opencode_siblings_are_unaffected():
    """Guard against a blanket edit to all 19 manifest entries."""
    from providers.registries.opencode_go import OpenCodeGoModelRegistry

    registry = OpenCodeGoModelRegistry()
    for name in ("glm-5.2", "deepseek-v4-pro", "qwen3.7-max"):
        assert registry.resolve(name).supports_temperature is True, name


# ---------------------------------------------------------------------------
# is_configured() -- the single implementation of the placeholder comparison
# that used to be copy-pasted into server.py and tools/shared/base_tool.py.
# ---------------------------------------------------------------------------


def test_is_configured_rejects_the_scaffolded_placeholder():
    """A .env left at its scaffolded value must not read as configured.

    ``.env.example`` ships ``OPENROUTER_API_KEY=your_openrouter_api_key_here``,
    so this is the common case, not a corner case. Before the refactor the
    comparison was inlined at four call sites; it now lives here only, which
    makes this the guard for all of them.
    """
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        placeholder = provider_cls.API_KEY_PLACEHOLDER
        if not placeholder:
            continue
        env = dict.fromkeys(provider_cls.gating_env_vars(), "real-value")
        env[provider_cls.API_KEY_ENV] = placeholder
        with patch.dict(os.environ, env, clear=True):
            assert provider_cls.is_configured() is False, provider_cls.__name__


def test_is_configured_accepts_a_real_key():
    """The negative test above is only meaningful if the positive one holds."""
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        gating = provider_cls.gating_env_vars()
        if not gating:
            continue
        with patch.dict(os.environ, {var: f"real-{var.lower()}" for var in gating}, clear=True):
            assert provider_cls.is_configured() is True, provider_cls.__name__


def test_is_configured_requires_every_gating_var():
    """Azure needs endpoint as well as key; Custom needs the URL."""
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        gating = provider_cls.gating_env_vars()
        if len(gating) < 2:
            continue
        for omitted in gating:
            env = {var: f"real-{var.lower()}" for var in gating if var != omitted}
            with patch.dict(os.environ, env, clear=True):
                assert provider_cls.is_configured() is False, (provider_cls.__name__, omitted)


def test_listmodels_custom_section_honours_the_allow_list():
    """CUSTOM_ALLOWED_MODELS must filter the listing, not just the call.

    ``OpenAICompatibleProvider`` has always enforced this variable at call
    time, so a name shown here that policy blocks is advertised and then
    rejected. The generic roster loop and the OpenRouter section already
    filtered; the bespoke Custom section did not.
    """
    import asyncio

    env = {
        "CUSTOM_API_URL": "http://localhost:11434/v1",
        "CUSTOM_ALLOWED_MODELS": "local-llama",
        "DEFAULT_MODEL": "local-llama",
    }
    with patch.dict(os.environ, env, clear=True):
        import utils.env as env_utils
        import utils.model_restrictions as model_restrictions

        env_utils.reload_env(env)
        model_restrictions._restriction_service = None
        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, CustomProvider)
        try:
            content = asyncio.run(ListModelsTool().execute({}))[0].text
        finally:
            model_restrictions._restriction_service = None
            env_utils.reload_env({})

    custom_section = content.split("## Custom/Local API")[1].split("\n## ")[0]
    assert "**Custom Models (policy restricted)**:" in custom_section
    assert "`local-llama`" in custom_section
    # Sibling aliases resolving to the same canonical model ARE permitted at
    # call time -- is_allowed() implements alias equivalence -- so the listing
    # must show them rather than under-advertise a usable model. Verified
    # directly: with a registered CUSTOM provider and a cold cache,
    # get_capabilities("ollama-llama") succeeds under CUSTOM_ALLOWED_MODELS=local-llama.
    assert "`ollama-llama`" in custom_section

    # This render disagrees with itself: three bullets above, two below.
    # Both numbers are pinned so a later reader cannot reconcile them by
    # "fixing" whichever one they meet first.
    #
    # Section = 3: list_aliases() yields the canonical plus both aliases, and
    # all three genuinely run under this allow-list (alias equivalence).
    #
    # Summary = 2: _collect_restricted_display_names enumerates the allow-list
    # itself rather than intersecting it with what the provider advertises, and
    # the allow-list has been WIDENED from {local-llama} to
    # {local-llama, gemma4:e4b} before this section is ever reached --
    # is_allowed() writes the resolved canonical back into the live set while
    # answering. Measured: the widening fires in get_available_models() at the
    # top of execute(), via list_models() -> one-arg is_allowed(), NOT in the
    # Custom-section loop below. Deleting that loop leaves the count at 2.
    #
    # So the mutating read-only path is list_models -> is_allowed, reached by
    # every caller of get_available_models -- not something this tool
    # introduced. It is pre-existing and load-bearing (deleting the write
    # fails test_error_listing_respects_env_restrictions and
    # test_openrouter_mixed_alias_and_full_names), hence a follow-up rather
    # than a fix here. Pinned so that follow-up has to move these numbers
    # deliberately.
    assert content.count("`gemma4:e4b` →") == 1, content
    assert content.count("`local-llama` →") == 1, content
    assert content.count("`ollama-llama` →") == 1, content
    assert "**Total Available Models**: 2" in content, content


# ---------------------------------------------------------------------------
# Allow-list typo check -- ask the provider, do not set-compare the manifest
# ---------------------------------------------------------------------------


def _warnings_for(provider, provider_type, allowed):
    """Run the startup typo check and return the warnings it emitted.

    ``validate_model_name`` consults the *singleton* restriction service via
    ``_ensure_model_allowed``, so the service under test has to be the
    singleton or the allow-list entries would be judged against a different
    policy than the one being validated.
    """
    import logging

    import utils.model_restrictions as model_restrictions

    service = model_restrictions.ModelRestrictionService()
    service.restrictions = {provider_type: set(allowed)}

    logger = logging.getLogger("utils.model_restrictions")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    previous = model_restrictions._restriction_service
    model_restrictions._restriction_service = service
    try:
        service.validate_against_known_models({provider_type: provider})
    finally:
        model_restrictions._restriction_service = previous
        logger.removeHandler(handler)

    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


def test_no_typo_warning_for_a_model_openrouter_actually_serves():
    """`vendor/model` is fabricated by _lookup_capabilities, so it is valid.

    Widening the startup validation loop to every provider would otherwise
    scold a correct config on every startup.
    """
    provider = OpenRouterProvider(api_key="test-key")
    assert provider.validate_model_name("vendor/some-model") is True

    assert _warnings_for(provider, ProviderType.OPENROUTER, {"vendor/some-model"}) == []


def test_typo_warning_still_fires_for_a_bad_openrouter_alias():
    """A misspelled curated alias has no "/" and is not fabricated.

    Skipping OpenRouter wholesale would give this signal up; asking the
    provider per entry keeps it.
    """
    provider = OpenRouterProvider(api_key="test-key")
    assert provider.validate_model_name("opuss") is False

    messages = _warnings_for(provider, ProviderType.OPENROUTER, {"opuss"})
    assert any("opuss" in m and "check for typos" in m for m in messages), messages


def test_typo_warning_still_fires_for_manifest_bound_providers():
    """The manifest is authoritative for OpenAI, so a typo must warn."""
    provider = OpenAIModelProvider(api_key="test-key")
    messages = _warnings_for(provider, ProviderType.OPENAI, {"gpt-nonexistent-typo"})
    assert any("gpt-nonexistent-typo" in m and "check for typos" in m for m in messages), messages


def test_typo_warning_fires_for_a_custom_endpoint_typo():
    """Custom's manifest IS exhaustive - _lookup_capabilities returns None for
    anything absent from custom_models.json, so validate_model_name is False
    and a typo'd CUSTOM_ALLOWED_MODELS must not pass silently.

    Regression guard: an earlier revision exempted Custom from this check on
    the false premise that it "serves whatever it is pointed at", which would
    have produced an empty custom listing with nothing in the logs.
    """
    provider = CustomProvider(api_key="", base_url="http://localhost:11434/v1")
    assert provider.validate_model_name("some-arbitrary-model") is False

    messages = _warnings_for(provider, ProviderType.CUSTOM, {"gemma4:e4-b"})
    assert any("gemma4:e4-b" in m and "check for typos" in m for m in messages), messages


def test_custom_base_url_env_is_named_not_positional():
    """Consumers ask for the base-URL variable by name, not by tuple index."""
    assert CustomProvider.BASE_URL_ENV == "CUSTOM_API_URL"
    assert CustomProvider.BASE_URL_ENV in CustomProvider.REQUIRED_ENV
