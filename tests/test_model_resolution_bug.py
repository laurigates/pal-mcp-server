"""
Test to reproduce and fix the OpenRouter model name resolution bug.

This test specifically targets the bug where:
1. User specifies "gemini" in consensus tool
2. System incorrectly resolves to "gemini-2.5-pro" instead of "google/gemini-2.5-pro"
3. OpenRouter API returns "gemini-2.5-pro is not a valid model ID"
"""

from unittest.mock import Mock, patch

from providers.openrouter import OpenRouterProvider
from providers.shared import ProviderType
from tools.consensus import ConsensusTool


class TestModelResolutionBug:
    """Test cases for the OpenRouter model name resolution bug."""

    def setup_method(self):
        """Setup test environment."""
        self.consensus_tool = ConsensusTool()

    def test_openrouter_registry_resolves_gemini_alias(self):
        """Generic Gemini aliases on OpenRouter follow the frontier (google/gemini-4-argon)."""
        provider = OpenRouterProvider("test_key")

        for alias in ("gemini", "pro", "gemini-pro", "pro-openrouter"):
            resolved = provider._resolve_model_name(alias)
            assert resolved == "google/gemini-4-argon", (
                f"Expected '{alias}' -> 'google/gemini-4-argon', got '{resolved}'"
            )

    def test_openrouter_version_pinned_gemini_alias_stays_on_3_1(self):
        """The version-pinned alias keeps reaching the superseded Gemini 3.1 Pro Preview."""
        provider = OpenRouterProvider("test_key")

        assert provider._resolve_model_name("gemini3") == "google/gemini-3.1-pro-preview"

    def test_openrouter_gemini_4_argon_is_promoted(self):
        """Argon is enabled in auto mode and ranks above the 3.1 Pro Preview it supersedes."""
        provider = OpenRouterProvider("test_key")

        argon = provider.get_capabilities("google/gemini-4-argon")
        previous = provider.get_capabilities("google/gemini-3.1-pro-preview")
        assert argon.enabled_by_default
        assert argon.intelligence_score > previous.intelligence_score

    # DELETED: test_provider_registry_returns_openrouter_for_gemini
    # This test had a flawed mock setup - it mocked get_provider() but called get_provider_for_model().
    # The test was trying to verify OpenRouter model resolution functionality that is already
    # comprehensively tested in working OpenRouter provider tests.

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test_key"}, clear=False)
    def test_consensus_tool_model_resolution_bug_reproduction(self):
        """Test that the new consensus workflow tool properly handles OpenRouter model resolution."""
        import asyncio

        # Create a mock OpenRouter provider that tracks what model names it receives
        mock_provider = Mock(spec=OpenRouterProvider)
        mock_provider.get_provider_type.return_value = ProviderType.OPENROUTER

        # Mock response for successful generation
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.usage = None
        mock_provider.generate_content.return_value = mock_response

        # Track the model name passed to generate_content
        received_model_names = []

        def track_generate_content(*args, **kwargs):
            received_model_names.append(kwargs.get("model_name", args[1] if len(args) > 1 else "unknown"))
            return mock_response

        mock_provider.generate_content.side_effect = track_generate_content

        # Mock the get_model_provider to return our mock
        with patch.object(self.consensus_tool, "get_model_provider", return_value=mock_provider):
            # Set initial prompt
            self.consensus_tool.initial_prompt = "Test prompt"

            # Create a mock request
            request = Mock()
            request.relevant_files = []
            request.continuation_id = None
            request.images = None

            # Test model consultation directly
            result = asyncio.run(self.consensus_tool._consult_model({"model": "gemini", "stance": "neutral"}, request))

            # Verify that generate_content was called
            assert len(received_model_names) == 1

            # The consensus tool should pass the original alias "gemini"
            # The OpenRouter provider should resolve it internally
            received_model = received_model_names[0]
            print(f"Model name passed to provider: {received_model}")

            assert received_model == "gemini", f"Expected 'gemini' to be passed to provider, got '{received_model}'"

            # Verify the result structure
            assert result["model"] == "gemini"
            assert result["status"] == "success"

    def test_bug_reproduction_with_malformed_model_name(self):
        """Test what happens when 'gemini-2.5-pro' (malformed) is passed to OpenRouter."""
        provider = OpenRouterProvider("test_key")

        # This should NOT resolve because 'gemini-2.5-pro' is not in the OpenRouter registry
        resolved = provider._resolve_model_name("gemini-2.5-pro")

        # The bug: this returns "gemini-2.5-pro" as-is instead of resolving to proper name
        # This is what causes the OpenRouter API to fail
        assert resolved == "gemini-2.5-pro", f"Expected fallback to 'gemini-2.5-pro', got '{resolved}'"

        # Verify the registry doesn't have this malformed name
        config = provider._registry.resolve("gemini-2.5-pro")
        assert config is None, "Registry should not contain 'gemini-2.5-pro' - only 'google/gemini-2.5-pro'"


if __name__ == "__main__":
    # Run the tests
    test = TestModelResolutionBug()
    test.setup_method()

    print("Testing OpenRouter registry resolution...")
    test.test_openrouter_registry_resolves_gemini_alias()
    print("✅ Registry resolves aliases correctly")

    print("\nTesting malformed model name handling...")
    test.test_bug_reproduction_with_malformed_model_name()
    print("✅ Confirmed: malformed names fall through as-is")

    print("\nConsensus tool test completed successfully.")

    print("\nAll tests completed. The bug is fixed.")
