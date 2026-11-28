"""Append-only conversation history (issue #195, Stage B, docs/prompt-caching.md).

Files render inside the turn that first listed them, so attaching a file appends
to the history instead of changing a files section ahead of every turn. When the
turns outgrow the budget, compaction drops old turns in one step, so the history
is rewritten once per compaction instead of on every call. Gates G2 and G7 in
``tests/test_prompt_prefix_stability.py`` check the request bytes end to end.
"""

from __future__ import annotations

from unittest.mock import patch

from utils.conversation_memory import (
    COMPACTION_KEEP_RATIO,
    NEW_INPUT_FILES_HEADER,
    OMITTED_TURNS_NOTE,
    ConversationTurn,
    ThreadContext,
    _compaction_start,
    build_conversation_history,
)
from utils.model_context import ModelContext


def _turn(role: str, content: str, files: list[str] | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, content=content, timestamp="2026-01-01T00:00:00Z", files=files)


def _history(turns, **kwargs) -> str:
    context = ThreadContext(
        thread_id="12345678-1234-1234-1234-123456789012",
        created_at="2026-01-01T00:00:00Z",
        last_updated_at="2026-01-01T00:00:00Z",
        tool_name="chat",
        turns=turns,
        initial_context={},
    )
    history, _ = build_conversation_history(context, ModelContext("flash"), **kwargs)
    return history


def _file(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_text(f"CONTENT-OF-{name}\n", encoding="utf-8")
    return str(path)


def test_each_file_renders_inside_the_turn_that_first_listed_it(tmp_path):
    a, b = _file(tmp_path, "a.py"), _file(tmp_path, "b.py")
    history = _history(
        [_turn("user", "q1", [a]), _turn("assistant", "r1"), _turn("user", "q2", [b]), _turn("assistant", "r2")]
    )

    turn_2, turn_3 = history.index("--- Turn 2 ("), history.index("--- Turn 3 (")
    assert history.index("CONTENT-OF-a.py") < turn_2
    assert turn_3 < history.index("CONTENT-OF-b.py")


def test_a_re_referenced_file_is_a_pointer_to_its_first_turn(tmp_path):
    a = _file(tmp_path, "a.py")
    history = _history([_turn("user", "q1", [a]), _turn("assistant", "r1"), _turn("user", "q2", [a])])

    assert history.count("CONTENT-OF-a.py") == 1
    assert f"[File {a}: see Turn 1]" in history


def test_attaching_a_file_appends_to_the_history(tmp_path):
    """The Stage B property: the history before the new turn does not change."""
    a, b = _file(tmp_path, "a.py"), _file(tmp_path, "b.py")
    earlier = [_turn("user", "q1", [a]), _turn("assistant", "r1")]
    before = _history(earlier)
    after = _history([*earlier, _turn("user", "q2", [b]), _turn("assistant", "r2")])

    shared_through = before.index("r1") + len("r1")
    assert before[:shared_through] == after[:shared_through]


def test_a_file_left_out_by_the_budget_is_a_marker_at_its_position(tmp_path):
    a, b = _file(tmp_path, "a.py"), _file(tmp_path, "b.py")
    with patch("utils.conversation_memory._plan_file_inclusion_by_size", return_value=([b], [a], 0)):
        history = _history([_turn("user", "q1", [a]), _turn("assistant", "r1"), _turn("user", "q2", [b])])

    assert "CONTENT-OF-a.py" not in history
    marker = history.index(f"[File {a} omitted")
    assert marker < history.index("--- Turn 2 (")
    assert "CONTENT-OF-b.py" in history


def test_the_new_input_files_follow_the_trailer(tmp_path):
    """Not after the last turn block: next request, that position holds this turn's block."""
    a = _file(tmp_path, "a.py")
    history = _history(
        [_turn("user", "q1"), _turn("assistant", "r1"), _turn("user", "CURRENT", [a])], latest_turn_is_input=True
    )

    assert "CURRENT" not in history
    assert history.index("=== END CONVERSATION HISTORY ===") < history.index(NEW_INPUT_FILES_HEADER)
    assert history.index(NEW_INPUT_FILES_HEADER) < history.index("CONTENT-OF-a.py")


def test_compaction_notes_the_dropped_turns_and_keeps_their_files(tmp_path):
    a = _file(tmp_path, "a.py")
    turns = [_turn("user", "q1", [a]), _turn("assistant", "r1"), _turn("user", "q2"), _turn("assistant", "r2")]
    with patch("utils.conversation_memory._compaction_start", return_value=2):
        history = _history(turns)

    assert f"[Note: Turns 1-2 {OMITTED_TURNS_NOTE}]" in history
    assert "--- Turn 1 (" not in history and "--- Turn 3 (" in history
    assert history.index("CONTENT-OF-a.py") < history.index("--- Turn 3 (")


class TestCompactionStart:
    def test_no_compaction_while_the_turns_fit(self):
        assert _compaction_start([10, 10, 10], budget=100) == 0

    def test_overflow_drops_old_turns_down_to_the_keep_ratio(self):
        start = _compaction_start([30, 30, 30, 30], budget=100)
        kept = sum([30, 30, 30, 30][start:])
        assert kept <= 100 * COMPACTION_KEEP_RATIO
        assert start == 2

    def test_newest_turn_that_fits_is_never_dropped(self):
        assert _compaction_start([300, 300, 300, 650], budget=1000) == 3
        assert _compaction_start([100] * 5 + [900], budget=1000) == 5

    def test_newest_turn_larger_than_the_budget_is_dropped(self):
        assert _compaction_start([100, 100, 1200], budget=1000) == 3

    def test_start_is_stable_between_compactions(self):
        """Request k+1 sees the turns of request k plus two more; start moves only at a compaction.

        With each request at 10% of the budget, a compaction frees room for four
        requests, so the start never moves on two consecutive requests. (Simulated:
        5 moves over 30 requests. From ~30% per request on, 60% hysteresis no
        longer spaces compactions out; docs/prompt-caching.md, Stage B.)
        """
        tokens = [5] * 60
        starts = [_compaction_start(tokens[:n], budget=100) for n in range(2, 61, 2)]
        moves = [i for i, (a, b) in enumerate(zip(starts, starts[1:], strict=False)) if a != b]

        assert starts == sorted(starts)
        assert moves, "the sequence must overflow, or the test measures nothing"
        assert not any(b == a + 1 for a, b in zip(moves, moves[1:], strict=False))

    def test_without_hysteresis_the_start_would_move_every_request(self):
        """Control for the test above: a newest-first fill moves the start on every request after overflow."""

        def newest_first_start(tokens, budget):
            total, start = 0, len(tokens)
            for index in range(len(tokens) - 1, -1, -1):
                if total + tokens[index] > budget:
                    break
                total += tokens[index]
                start = index
            return start

        tokens = [5] * 60
        starts = [newest_first_start(tokens[:n], 100) for n in range(2, 61, 2)]
        overflowed = [s for s in starts if s > 0]
        assert len(set(overflowed)) == len(overflowed)
