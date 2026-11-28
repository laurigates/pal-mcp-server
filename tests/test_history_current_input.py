"""The continuation history carries the current request once (issue #194, Stage A).

Both history builders record the caller's turn first, so the files it attaches
reach the model on that turn, and then append the request under
``=== NEW USER INPUT ===``. ``latest_turn_is_input`` keeps the recorded turn's
files in the history and leaves its text to the NEW USER INPUT slot.

The header carries no turn counter: a counter there is the first byte that
changes between consecutive requests, so no provider cache can reuse anything
after it.
"""

from __future__ import annotations

from utils.conversation_memory import ConversationTurn, ThreadContext, build_conversation_history
from utils.model_context import ModelContext

THREAD_ID = "12345678-1234-1234-1234-123456789012"


def _context(turns: list[ConversationTurn]) -> ThreadContext:
    return ThreadContext(
        thread_id=THREAD_ID,
        created_at="2026-01-01T00:00:00Z",
        last_updated_at="2026-01-01T00:00:00Z",
        tool_name="chat",
        turns=turns,
        initial_context={},
    )


def _turn(role: str, content: str, files: list[str] | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, timestamp="2026-01-01T00:00:00Z", files=files)


def _history(turns, **kwargs) -> str:
    history, _ = build_conversation_history(_context(turns), ModelContext("flash"), **kwargs)
    return history


def test_current_input_is_left_out_of_the_turns_but_its_files_are_embedded(tmp_path):
    attached = tmp_path / "attached.py"
    attached.write_text("ATTACHED-ON-THIS-TURN = 1\n", encoding="utf-8")
    turns = [
        _turn("user", "first question"),
        _turn("assistant", "first answer"),
        _turn("user", "CURRENT QUESTION", files=[str(attached)]),
    ]

    history = _history(turns, latest_turn_is_input=True)

    assert "CURRENT QUESTION" not in history
    assert "ATTACHED-ON-THIS-TURN" in history
    assert "--- Turn 2 (" in history and "--- Turn 3 (" not in history
    assert "This is turn 3 of the conversation" in history


def test_without_the_flag_every_recorded_turn_is_rendered():
    turns = [_turn("user", "first question"), _turn("assistant", "first answer"), _turn("user", "LAST QUESTION")]

    history = _history(turns)

    assert "LAST QUESTION" in history
    assert "This is turn 4 of the conversation" in history


def test_the_flag_leaves_a_trailing_assistant_turn_alone():
    turns = [_turn("user", "first question"), _turn("assistant", "LAST ANSWER")]

    history = _history(turns, latest_turn_is_input=True)

    assert "LAST ANSWER" in history
    assert "This is turn 3 of the conversation" in history


def test_header_has_no_turn_counter():
    turns = [_turn("user", "q"), _turn("assistant", "a")]

    header = _history(turns).split("You are continuing this conversation thread", 1)[0]

    assert header.splitlines()[1:] == [f"Thread: {THREAD_ID}", "Tool: chat"]


def test_consecutive_histories_share_everything_through_the_last_turn():
    """The prefix-cache property the header change buys: request j+1 extends request j's turns."""
    first = [_turn("user", "q1"), _turn("assistant", "a1"), _turn("user", "q2")]
    second = [*first, _turn("assistant", "a2"), _turn("user", "q3")]

    earlier = _history(first, latest_turn_is_input=True)
    later = _history(second, latest_turn_is_input=True)

    shared_through = earlier.index("a1") + len("a1")
    assert earlier[:shared_through] == later[:shared_through]
