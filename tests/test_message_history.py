"""Stage D prototype: history as role messages, appended request by request (#197, #260).

GPT-5.6+ caches only the whole previous request as a prefix, cut at a message
boundary: a probe on gpt-6-luna (2026-11-28) read 0 when the last message changed
or grew, 0 at earlier message boundaries, and 14,435 of 14,448 tokens when the
previous request was resent unchanged with one message appended. So request j+1
must be request j's messages, byte for byte, plus the reply and the new input.

PAL keeps each turn's user message as it was sent (``sent_prompt`` on the reply's
metadata) and replays it, instead of rendering the history into one message.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from providers.openai import OpenAIModelProvider
from utils.conversation_memory import ConversationTurn, ThreadContext
from utils.message_history import SENT_PROMPT_KEY, build_history_messages, message_history_enabled

NOW = datetime.now(timezone.utc).isoformat()


def _turn(role, content, *, sent=None, tool="chat"):
    metadata = {SENT_PROMPT_KEY: sent} if sent is not None else None
    return ConversationTurn(role=role, content=content, timestamp=NOW, tool_name=tool, model_metadata=metadata)


def _context(turns, parent=None):
    return ThreadContext(
        thread_id="t-1",
        parent_thread_id=parent,
        created_at=NOW,
        last_updated_at=NOW,
        tool_name="chat",
        turns=turns,
        initial_context={},
    )


class TestBuildHistoryMessages:
    def test_replays_each_sent_prompt_and_reply_in_order(self):
        context = _context(
            [
                _turn("user", "q1"),
                _turn("assistant", "a1", sent="SENT q1 + files"),
                _turn("user", "q2"),
                _turn("assistant", "a2", sent="SENT q2"),
                _turn("user", "q3"),  # the request being answered
            ]
        )
        assert build_history_messages(context, tool_name="chat") == [
            {"role": "user", "content": "SENT q1 + files"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "SENT q2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_consecutive_requests_are_strict_appends(self):
        """The property the cache needs: history j+1 starts with history j plus j's request."""
        turns = [_turn("user", "q1"), _turn("assistant", "a1", sent="S1"), _turn("user", "q2")]
        first = build_history_messages(_context(turns), tool_name="chat") + [{"role": "user", "content": "S2"}]
        turns += [_turn("assistant", "a2", sent="S2"), _turn("user", "q3")]
        second = build_history_messages(_context(turns), tool_name="chat")
        assert second[: len(first)] == first

    def test_a_reply_without_its_sent_prompt_falls_back(self):
        context = _context([_turn("user", "q1"), _turn("assistant", "a1"), _turn("user", "q2")])
        assert build_history_messages(context, tool_name="chat") is None

    def test_a_cross_tool_thread_falls_back(self):
        context = _context(
            [_turn("user", "q1"), _turn("assistant", "a1", sent="S1", tool="debug"), _turn("user", "q2")]
        )
        assert build_history_messages(context, tool_name="chat") is None

    def test_a_chained_thread_falls_back(self):
        context = _context([_turn("user", "q1"), _turn("assistant", "a1", sent="S1")], parent="t-0")
        assert build_history_messages(context, tool_name="chat") is None

    def test_a_thread_with_no_reply_yet_falls_back(self):
        assert build_history_messages(_context([_turn("user", "q1")]), tool_name="chat") is None


class TestEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("PAL_MESSAGE_HISTORY", raising=False)
        assert not message_history_enabled(OpenAIModelProvider("k"))

    def test_on_for_a_provider_that_supports_it(self, monkeypatch):
        monkeypatch.setenv("PAL_MESSAGE_HISTORY", "true")
        assert message_history_enabled(OpenAIModelProvider("k"))

    def test_off_for_a_provider_that_does_not(self, monkeypatch):
        from providers.gemini import GeminiModelProvider

        monkeypatch.setenv("PAL_MESSAGE_HISTORY", "true")
        assert not message_history_enabled(GeminiModelProvider("k"))


class TestProviderRequest:
    @pytest.mark.parametrize("model", ["gpt-5", "gpt-6-luna"])  # Chat Completions, Responses
    def test_history_messages_sit_between_system_and_input(self, model):
        history = [{"role": "user", "content": "S1"}, {"role": "assistant", "content": "a1"}]
        request = OpenAIModelProvider("k")._build_request(
            prompt="S2",
            model_name=model,
            system_prompt="SYS",
            temperature=1.0,
            max_output_tokens=None,
            history_messages=history,
        )
        items = request["params"].get("messages") or request["params"]["input"]

        def text(item):
            content = item["content"]
            return content if isinstance(content, str) else content[0]["text"]

        assert [(i["role"], text(i)) for i in items] == [
            ("system" if "messages" in request["params"] else "user", "SYS"),
            ("user", "S1"),
            ("assistant", "a1"),
            ("user", "S2"),
        ]
