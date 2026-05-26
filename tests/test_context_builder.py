"""
Unit tests for ``utils.context_builder.build_enhanced_prompt``.

Verifies the prompt-assembly contract that the previous inline block
in ``server.reconstruct_thread_context`` implemented: ordering
(history → user input → follow-up), token budget clamping, and the
"no history" short-circuit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from utils.context_builder import build_enhanced_prompt


def _make_model_context(*, content_tokens: int = 1000):
    """Stub ModelContext exposing only the surface build_enhanced_prompt needs."""
    model_context = MagicMock()
    model_context.model_name = "test-model"
    allocation = MagicMock()
    allocation.content_tokens = content_tokens
    model_context.calculate_token_allocation.return_value = allocation
    return model_context


def _make_thread_context():
    """A lightweight ThreadContext-shaped MagicMock — only the type tag matters here."""
    return MagicMock(name="thread_context")


class TestBuildEnhancedPrompt:
    """Pin the assembly order and budget math."""

    def test_returns_tuple_of_str_and_int(self):
        """Public contract: (enhanced_prompt: str, remaining_tokens: int)."""
        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("prior turns", 100),
        ):
            enhanced_prompt, remaining = build_enhanced_prompt(
                thread_context=_make_thread_context(),
                model_context=_make_model_context(content_tokens=500),
                user_prompt="hi",
                follow_up_instructions="follow up please",
            )

        assert isinstance(enhanced_prompt, str)
        assert isinstance(remaining, int)

    def test_history_appears_before_new_input_and_follow_ups(self):
        """Order: history → NEW USER INPUT marker → user input → follow-up."""
        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("=== HISTORY ===\nturn1", 50),
        ):
            enhanced_prompt, _ = build_enhanced_prompt(
                thread_context=_make_thread_context(),
                model_context=_make_model_context(),
                user_prompt="my-question",
                follow_up_instructions="FOLLOWUP",
            )

        history_pos = enhanced_prompt.index("=== HISTORY ===")
        marker_pos = enhanced_prompt.index("=== NEW USER INPUT ===")
        user_pos = enhanced_prompt.index("my-question")
        follow_pos = enhanced_prompt.index("FOLLOWUP")

        assert history_pos < marker_pos < user_pos < follow_pos

    def test_no_history_omits_marker(self):
        """Empty history → straight ``{user}\\n\\n{follow_up}`` shape, no marker."""
        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("", 0),
        ):
            enhanced_prompt, _ = build_enhanced_prompt(
                thread_context=_make_thread_context(),
                model_context=_make_model_context(),
                user_prompt="fresh-question",
                follow_up_instructions="FOLLOWUP",
            )

        assert "=== NEW USER INPUT ===" not in enhanced_prompt
        assert enhanced_prompt.startswith("fresh-question")
        assert enhanced_prompt.endswith("FOLLOWUP")

    def test_remaining_tokens_subtracts_history(self):
        """remaining = content_tokens − history_tokens, clamped to ≥0."""
        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("history", 300),
        ):
            _, remaining = build_enhanced_prompt(
                thread_context=_make_thread_context(),
                model_context=_make_model_context(content_tokens=1000),
                user_prompt="x",
                follow_up_instructions="y",
            )

        assert remaining == 700

    def test_remaining_tokens_clamped_to_zero_when_history_exceeds_budget(self):
        """If history consumed > content budget, remaining is clamped to 0, not negative."""
        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("history", 2000),
        ):
            _, remaining = build_enhanced_prompt(
                thread_context=_make_thread_context(),
                model_context=_make_model_context(content_tokens=1000),
                user_prompt="x",
                follow_up_instructions="y",
            )

        assert remaining == 0

    def test_passes_contexts_through_to_build_conversation_history(self):
        """build_enhanced_prompt should forward both contexts unmodified to history-builder."""
        thread = _make_thread_context()
        model = _make_model_context()

        with patch(
            "utils.context_builder.build_conversation_history",
            return_value=("h", 10),
        ) as bch:
            build_enhanced_prompt(
                thread_context=thread,
                model_context=model,
                user_prompt="u",
                follow_up_instructions="f",
            )

        bch.assert_called_once_with(thread, model)
