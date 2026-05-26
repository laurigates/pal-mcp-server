"""
Enhanced-prompt assembly for conversation continuation.

Wraps the history-build → follow-up injection → token-budget block
that ``reconstruct_thread_context`` previously inlined. Exposes a
single ``build_enhanced_prompt`` helper so the orchestrator can stay
focused on flow control rather than string formatting.

The returned tuple matches the shape the caller expects:

* ``enhanced_prompt`` — final prompt string with conversation history,
  the latest user input, and any follow-up instructions.
* ``remaining_tokens_for_files`` — how much of the content-token
  budget is left for file payloads after the conversation history
  has consumed its share. Always non-negative.

Pure function: no side effects, no logging side effects, no
``arguments`` mutation. Callers assemble the resulting strings into
the request dict themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.conversation_memory import build_conversation_history

if TYPE_CHECKING:
    from utils.conversation_memory import ThreadContext
    from utils.model_context import ModelContext


def build_enhanced_prompt(
    *,
    thread_context: ThreadContext,
    model_context: ModelContext,
    user_prompt: str,
    follow_up_instructions: str,
) -> tuple[str, int]:
    """
    Build the enhanced prompt and compute the remaining token budget.

    Args:
        thread_context: The loaded ``ThreadContext`` for the
            continuation. Used to build conversation history with
            model-specific token limits.
        model_context: The ``ModelContext`` chosen for this
            reconstruction (see ``resolve_model_for_context``).
        user_prompt: The current turn's user input. May be empty.
        follow_up_instructions: Pre-computed follow-up instructions
            (from ``get_follow_up_instructions``).

    Returns:
        Tuple of ``(enhanced_prompt, remaining_tokens_for_files)``.
        ``remaining_tokens_for_files`` is clamped to ``>= 0``.
    """
    conversation_history, conversation_tokens = build_conversation_history(thread_context, model_context)

    if conversation_history:
        enhanced_prompt = f"{conversation_history}\n\n=== NEW USER INPUT ===\n{user_prompt}\n\n{follow_up_instructions}"
    else:
        enhanced_prompt = f"{user_prompt}\n\n{follow_up_instructions}"

    token_allocation = model_context.calculate_token_allocation()
    remaining_tokens = token_allocation.content_tokens - conversation_tokens
    remaining_tokens = max(0, remaining_tokens)

    return enhanced_prompt, remaining_tokens
