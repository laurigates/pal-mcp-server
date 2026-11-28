"""Stage D prototype: send a thread's history as role messages (#197, #260).

GPT-5.6+ caches the whole previous request and reuses it only when the next
request starts with it, message for message: changing or extending the last
message, or diverging after an earlier message boundary, reads nothing (probe on
gpt-6-luna, 2026-11-28; docs/prompt-caching.md). The rendered history (Stages A
and B) puts every turn in one user message that grows on each request, so those
models never read their cache and pay the write premium on every call.

Here each turn's user message is kept exactly as it was sent (``sent_prompt`` on
the reply's ``model_metadata``) and replayed, followed by the reply, so request
j+1 is request j plus two messages. Behind ``PAL_MESSAGE_HISTORY=true`` and only
for providers that declare ``SUPPORTS_MESSAGE_HISTORY``; anything this cannot
replay (a reply without its sent prompt, a cross-tool or chained thread) falls
back to the rendered history.
"""

from __future__ import annotations

import os
from typing import Any

from utils.conversation_memory import ThreadContext

MESSAGE_HISTORY_ENV = "PAL_MESSAGE_HISTORY"
#: model_metadata key holding the user message exactly as it was sent for that reply.
SENT_PROMPT_KEY = "sent_prompt"


def message_history_enabled(provider: Any) -> bool:
    if os.environ.get(MESSAGE_HISTORY_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return bool(getattr(provider, "SUPPORTS_MESSAGE_HISTORY", False))


def build_history_messages(context: ThreadContext, *, tool_name: str) -> list[dict[str, str]] | None:
    """The thread's earlier exchanges as ``user``/``assistant`` messages, or None to fall back.

    The last user turn (the request being answered) is not included: the caller
    sends it as the new input.
    """
    if context.parent_thread_id:
        return None
    replies = [turn for turn in context.turns if turn.role == "assistant"]
    if not replies:
        return None
    messages: list[dict[str, str]] = []
    for reply in replies:
        sent = (reply.model_metadata or {}).get(SENT_PROMPT_KEY)
        if not isinstance(sent, str) or reply.tool_name != tool_name:
            return None
        messages.append({"role": "user", "content": sent})
        messages.append({"role": "assistant", "content": reply.content})
    return messages
