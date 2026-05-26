"""Tests for OpenRouter store parameter handling in responses endpoint.

Regression tests for GitHub Issue #348: OpenAI "store" parameter validation error
for certain models via OpenRouter.

OpenRouter's /responses endpoint rejects store:true via Zod validation. This is an
endpoint-level limitation, not model-specific. These tests verify that:
- OpenRouter provider omits the store parameter
- Direct OpenAI provider includes store: true
"""

import unittest
from unittest.mock import Mock

from providers.openai_compatible import OpenAICompatibleProvider
from providers.shared import ProviderType


def _make_capabilities():
    """Build a capabilities object that routes through the responses endpoint."""
    capabilities = Mock()
    capabilities.default_reasoning_effort = "high"
    capabilities.use_openai_response_api = True
    capabilities.supports_images = False
    capabilities.get_effective_temperature = lambda t: t
    return capabilities


class MockOpenRouterProvider(OpenAICompatibleProvider):
    """Mock provider that simulates OpenRouter behavior."""

    FRIENDLY_NAME = "OpenRouter Test"

    def get_provider_type(self):
        return ProviderType.OPENROUTER

    def get_capabilities(self, model_name):
        return _make_capabilities()

    def get_all_model_capabilities(self):
        return {}

    def validate_model_name(self, model_name):
        return True

    def validate_parameters(self, model_name, temperature, **kwargs):
        return None

    def list_models(self, **kwargs):
        return ["openai/gpt-5-pro", "openai/gpt-5.1-codex"]


class MockOpenAIProvider(OpenAICompatibleProvider):
    """Mock provider that simulates direct OpenAI behavior."""

    FRIENDLY_NAME = "OpenAI Test"

    def get_provider_type(self):
        return ProviderType.OPENAI

    def get_capabilities(self, model_name):
        return _make_capabilities()

    def get_all_model_capabilities(self):
        return {}

    def validate_model_name(self, model_name):
        return True

    def validate_parameters(self, model_name, temperature, **kwargs):
        return None

    def list_models(self, **kwargs):
        return ["gpt-5-pro", "gpt-5.1-codex"]


class TestStoreParameterHandling(unittest.TestCase):
    """Test store parameter is conditionally included based on provider type.

    Wave B moved the responses-endpoint shape from a dedicated
    ``_generate_with_responses_endpoint`` method into ``_build_request``.
    These tests now inspect the dict ``_build_request`` returns directly.

    **Feature: openrouter-store-parameter-fix, Property 1: OpenRouter requests omit store parameter**
    **Feature: openrouter-store-parameter-fix, Property 2: Direct OpenAI requests include store parameter**
    """

    def test_openrouter_responses_omits_store_parameter(self):
        """OpenRouter provider must omit ``store`` from responses-endpoint params.

        OpenRouter's /responses endpoint rejects store:true via Zod validation (Issue #348).
        """
        provider = MockOpenRouterProvider("test-key")
        request = provider._build_request(
            prompt="test",
            model_name="openai/gpt-5-pro",
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=None,
        )

        assert request["endpoint"] == "responses", "OpenRouter should route via the responses endpoint"
        assert "store" not in request["params"], "OpenRouter requests must NOT include 'store' parameter"

    def test_openai_responses_includes_store_parameter(self):
        """Direct OpenAI provider must include ``store: True`` in responses-endpoint params."""
        provider = MockOpenAIProvider("test-key")
        request = provider._build_request(
            prompt="test",
            model_name="gpt-5-pro",
            system_prompt=None,
            temperature=0.7,
            max_output_tokens=None,
        )

        assert request["endpoint"] == "responses", "OpenAI should route via the responses endpoint"
        assert "store" in request["params"], "OpenAI requests should include 'store' parameter"
        assert request["params"]["store"] is True, "OpenAI requests should have store=True"


if __name__ == "__main__":
    unittest.main()
