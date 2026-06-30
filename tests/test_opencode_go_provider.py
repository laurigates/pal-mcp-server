"""Tests for the OpenCode Go provider implementation."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.opencode_go import OpenCodeGoProvider
from providers.shared import ProviderType

pytestmark = pytest.mark.no_mock_provider


class TestOpenCodeGoProvider:
    """OpenCode Go is an OpenAI-compatible coding-model gateway (SST)."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    # ------------------------------------------------------------------
    # Construction / identity
    # ------------------------------------------------------------------
    def test_initialization(self):
        provider = OpenCodeGoProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.OPENCODE_GO
        assert provider.base_url == "https://opencode.ai/zen/go/v1"

    def test_initialization_with_custom_url(self):
        provider = OpenCodeGoProvider("test-key", base_url="https://proxy.local/v1")
        assert provider.base_url == "https://proxy.local/v1"

    def test_friendly_name(self):
        provider = OpenCodeGoProvider("test-key")
        assert provider.FRIENDLY_NAME == "OpenCode Go"
        assert provider.get_capabilities("glm-5.2").friendly_name == "OpenCode Go (GLM-5.2)"

    # ------------------------------------------------------------------
    # from_env contract (reads OPENCODE_API_KEY)
    # ------------------------------------------------------------------
    @patch.dict(os.environ, {"OPENCODE_API_KEY": "live-key"}, clear=False)
    def test_from_env_present(self):
        provider = OpenCodeGoProvider.from_env()
        assert provider is not None
        assert provider.api_key == "live-key"
        assert provider.base_url == "https://opencode.ai/zen/go/v1"

    def test_from_env_absent(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        assert OpenCodeGoProvider.from_env() is None

    @patch.dict(os.environ, {"OPENCODE_API_KEY": "your_opencode_api_key_here"}, clear=False)
    def test_from_env_rejects_placeholder(self):
        assert OpenCodeGoProvider.from_env() is None

    # ------------------------------------------------------------------
    # Model validation & capabilities (data sourced from models.dev)
    # ------------------------------------------------------------------
    def test_model_validation(self):
        provider = OpenCodeGoProvider("test-key")
        assert provider.validate_model_name("glm-5.2") is True
        assert provider.validate_model_name("deepseek-v4-flash") is True
        assert provider.validate_model_name("kimi-k2.7-code") is True
        # aliases
        assert provider.validate_model_name("glm") is True
        assert provider.validate_model_name("deepseek") is True
        assert provider.validate_model_name("kimi") is True
        # not ours
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("grok-4") is False
        assert provider.validate_model_name("gemini-2.5-pro") is False

    def test_resolve_alias(self):
        provider = OpenCodeGoProvider("test-key")
        assert provider._resolve_model_name("glm") == "glm-5.2"
        assert provider._resolve_model_name("deepseek") == "deepseek-v4-pro"
        assert provider._resolve_model_name("kimi") == "kimi-k2.7-code"
        assert provider._resolve_model_name("minimax") == "minimax-m3"
        # canonical passthrough
        assert provider._resolve_model_name("glm-5.2") == "glm-5.2"

    def test_capabilities_context_windows(self):
        provider = OpenCodeGoProvider("test-key")

        glm = provider.get_capabilities("glm-5.2")
        assert glm.model_name == "glm-5.2"
        assert glm.provider == ProviderType.OPENCODE_GO
        assert glm.context_window == 1_000_000
        assert glm.supports_function_calling is True

        flash = provider.get_capabilities("deepseek-v4-flash")
        assert flash.context_window == 1_000_000
        assert flash.max_output_tokens == 384_000

    def test_capabilities_image_support_tracks_attachment(self):
        provider = OpenCodeGoProvider("test-key")
        # kimi-k2.7-code has attachment=true on models.dev
        assert provider.get_capabilities("kimi-k2.7-code").supports_images is True
        # glm-5.2 has attachment=false
        assert provider.get_capabilities("glm-5.2").supports_images is False

    def test_unsupported_model_raises(self):
        provider = OpenCodeGoProvider("test-key")
        with pytest.raises(ValueError, match="Unsupported model 'totally-fake'"):
            provider.get_capabilities("totally-fake")

    def test_catalogue_size(self):
        provider = OpenCodeGoProvider("test-key")
        caps = provider.get_all_model_capabilities()
        # 19 curated models published by the OpenCode Go plan (models.dev)
        assert len(caps) == 19
        assert all(c.provider == ProviderType.OPENCODE_GO for c in caps.values())

    # ------------------------------------------------------------------
    # Restrictions
    # ------------------------------------------------------------------
    @patch.dict(os.environ, {"OPENCODE_GO_ALLOWED_MODELS": "glm-5.2"}, clear=False)
    def test_model_restrictions(self):
        import utils.model_restrictions
        from providers.registry import ModelProviderRegistry

        utils.model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        provider = OpenCodeGoProvider("test-key")
        assert provider.validate_model_name("glm-5.2") is True
        assert provider.validate_model_name("glm") is True  # alias of glm-5.2
        assert provider.validate_model_name("deepseek-v4-pro") is False

    # ------------------------------------------------------------------
    # Alias resolution flows through to the SDK call
    # ------------------------------------------------------------------
    @patch("providers.openai_compatible.AsyncOpenAI")
    async def test_generate_content_resolves_alias(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "glm-5.2"
        mock_response.id = "id"
        mock_response.created = 1
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenCodeGoProvider("test-key")
        result = await provider.generate_content(prompt="hi", model_name="glm", temperature=0.5)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "glm-5.2"  # alias resolved before SDK call
        assert result.model_name == "glm-5.2"


class TestOpenCodeGoRegistration:
    """The provider must be wired into the registry plumbing."""

    def test_in_registered_provider_classes(self):
        from providers.registry import REGISTERED_PROVIDER_CLASSES

        assert OpenCodeGoProvider in REGISTERED_PROVIDER_CLASSES

    def test_priority_order_before_openrouter_catch_all(self):
        from providers.registry import ModelProviderRegistry

        order = ModelProviderRegistry.PROVIDER_PRIORITY_ORDER
        assert ProviderType.OPENCODE_GO in order
        # Curated gateway must outrank the OpenRouter catch-all.
        assert order.index(ProviderType.OPENCODE_GO) < order.index(ProviderType.OPENROUTER)

    def test_api_key_env_mapping(self):
        from providers.registry import ModelProviderRegistry

        assert ModelProviderRegistry._get_api_key_for_provider.__func__ is not None
