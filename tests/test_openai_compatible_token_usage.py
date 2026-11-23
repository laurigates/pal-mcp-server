"""Tests for OpenAI-compatible provider token usage extraction."""

import unittest
from unittest.mock import Mock

from providers.openai_compatible import OpenAICompatibleProvider


class TestOpenAICompatibleTokenUsage(unittest.TestCase):
    """Test OpenAI-compatible provider token usage handling."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class TestProvider(OpenAICompatibleProvider):
            FRIENDLY_NAME = "Test"
            MODEL_CAPABILITIES = {"test-model": {"context_window": 4096}}

            def get_capabilities(self, model_name):
                return Mock()

            def get_provider_type(self):
                return Mock()

            def validate_model_name(self, model_name):
                return True

            def list_models(self, **kwargs):
                return ["test-model"]

        self.provider = TestProvider("test-key")

    def test_extract_usage_with_valid_tokens(self):
        """Test token extraction with valid token counts."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 150

        usage = self.provider._extract_usage(response)

        self.assertEqual(usage["input_tokens"], 100)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 150)

    def test_extract_usage_with_none_prompt_tokens(self):
        """Test token extraction when prompt_tokens is None (regression test for bug)."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = None  # This was causing crashes
        response.usage.completion_tokens = 50
        response.usage.total_tokens = None

        usage = self.provider._extract_usage(response)

        # Should default to 0 when None
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 0)

    def test_extract_usage_with_none_completion_tokens(self):
        """Test token extraction when completion_tokens is None (regression test for bug)."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = None  # This was causing crashes
        response.usage.total_tokens = None

        usage = self.provider._extract_usage(response)

        self.assertEqual(usage["input_tokens"], 100)
        # Should default to 0 when None
        self.assertEqual(usage["output_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_extract_usage_with_all_none_tokens(self):
        """Test token extraction when all token counts are None."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = None
        response.usage.completion_tokens = None
        response.usage.total_tokens = None

        usage = self.provider._extract_usage(response)

        # Should default to 0 for all when None
        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_extract_usage_without_usage(self):
        """Test token extraction when response has no usage."""
        response = Mock(spec=[])  # No usage attribute

        usage = self.provider._extract_usage(response)

        # Should return empty dict
        self.assertEqual(usage, {})

    def test_extract_usage_with_zero_tokens(self):
        """Test token extraction with zero token counts."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = 0
        response.usage.completion_tokens = 0
        response.usage.total_tokens = 0

        usage = self.provider._extract_usage(response)

        self.assertEqual(usage["input_tokens"], 0)
        self.assertEqual(usage["output_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)

    def test_alternative_token_format_with_none(self):
        """Test alternative token format (input_tokens/output_tokens) with None values."""
        # This tests the other code path in generate_content_openai_responses
        # Simulate a response with input_tokens/output_tokens attributes that could be None
        response = Mock()
        response.input_tokens = None  # This was causing crashes
        response.output_tokens = 50

        # Test the pattern: getattr(response, "input_tokens", 0) or 0
        input_tokens = getattr(response, "input_tokens", 0) or 0
        output_tokens = getattr(response, "output_tokens", 0) or 0

        # Should not crash and should handle None gracefully
        self.assertEqual(input_tokens, 0)
        self.assertEqual(output_tokens, 50)

        # Test that addition works
        total = input_tokens + output_tokens
        self.assertEqual(total, 50)

    # -- prompt-cache fields (issue #193) ---------------------------------

    def _chat_usage(self, **details):
        usage = Mock(spec=["prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"])
        usage.prompt_tokens = 1000
        usage.completion_tokens = 20
        usage.total_tokens = 1020
        usage.prompt_tokens_details = Mock(spec=list(details))
        for key, value in details.items():
            setattr(usage.prompt_tokens_details, key, value)
        response = Mock(spec=["usage"])
        response.usage = usage
        return response

    def test_chat_reads_cached_and_cache_write_tokens(self):
        usage = self.provider._extract_usage(self._chat_usage(cached_tokens=768, cache_write_tokens=128))

        self.assertEqual(usage["cached_input_tokens"], 768)
        self.assertEqual(usage["cache_write_input_tokens"], 128)
        self.assertEqual(usage["input_tokens"], 1000)

    def test_chat_cached_zero_is_recorded_not_dropped(self):
        usage = self.provider._extract_usage(self._chat_usage(cached_tokens=0))

        self.assertEqual(usage["cached_input_tokens"], 0)
        self.assertNotIn("cache_write_input_tokens", usage)

    def test_chat_absent_cache_fields_stay_absent(self):
        """A response without the field must not claim a miss: coverage is measured from absence."""
        usage = self.provider._extract_usage(self._chat_usage())

        self.assertNotIn("cached_input_tokens", usage)
        self.assertNotIn("cache_write_input_tokens", usage)

    def test_chat_falls_back_to_deepseek_prompt_cache_hit_tokens(self):
        response = Mock(spec=["usage"])
        response.usage = Mock(spec=["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens"])
        response.usage.prompt_tokens = 500
        response.usage.completion_tokens = 10
        response.usage.total_tokens = 510
        response.usage.prompt_cache_hit_tokens = 384

        usage = self.provider._extract_usage(response)

        self.assertEqual(usage["cached_input_tokens"], 384)

    def test_bare_mock_usage_adds_no_cache_fields(self):
        """Existing tests build usage from bare Mock(); attribute auto-creation must not leak in."""
        response = Mock()
        response.usage = Mock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15

        usage = self.provider._extract_usage(response)

        self.assertEqual(usage, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    def test_responses_extractor_reads_input_and_cached_tokens(self):
        response = Mock(spec=["usage"])
        response.usage = Mock(spec=["input_tokens", "output_tokens", "total_tokens", "input_tokens_details"])
        response.usage.input_tokens = 2048
        response.usage.output_tokens = 64
        response.usage.total_tokens = 2112
        response.usage.input_tokens_details = Mock(spec=["cached_tokens", "cache_write_tokens"])
        response.usage.input_tokens_details.cached_tokens = 1536
        response.usage.input_tokens_details.cache_write_tokens = 256

        usage = self.provider._extract_responses_usage(response)

        self.assertEqual(
            usage,
            {
                "input_tokens": 2048,
                "output_tokens": 64,
                "total_tokens": 2112,
                "cached_input_tokens": 1536,
                "cache_write_input_tokens": 256,
            },
        )

    def test_responses_extractor_without_total_sums_input_and_output(self):
        response = Mock(spec=["usage"])
        response.usage = Mock(spec=["input_tokens", "output_tokens"])
        response.usage.input_tokens = 30
        response.usage.output_tokens = 12

        usage = self.provider._extract_responses_usage(response)

        self.assertEqual(usage, {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42})


if __name__ == "__main__":
    unittest.main()
