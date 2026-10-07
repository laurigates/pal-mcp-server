"""Tests for per-thread JSONL conversation transcripts (utils.transcripts, issue #164)."""

import json
import logging
import os
import stat
import time

import pytest

import utils.conversation_memory as conversation_memory
from utils import transcripts
from utils.conversation_memory import ConversationTurn


@pytest.fixture
def state_dir(monkeypatch, tmp_path):
    path = tmp_path / "state"
    monkeypatch.setenv("PAL_STATE_DIR", str(path))
    monkeypatch.delenv("PAL_TRANSCRIPTS", raising=False)
    return path


def _read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_transcripts_dir_is_threads_under_state_dir(state_dir):
    assert transcripts.get_transcripts_dir() == state_dir / "threads"
    assert transcripts.transcript_path("abc") == state_dir / "threads" / "abc.jsonl"


def test_enabled_by_default(state_dir):
    assert transcripts.transcripts_enabled() is True


def test_create_thread_and_add_turn_write_header_and_turn_records(state_dir):
    thread_id = conversation_memory.create_thread("chat", {"prompt": "secret request params"})

    assert conversation_memory.add_turn(thread_id, "user", "What is 2+2?", files=["/a.py"], tool_name="chat")
    assert conversation_memory.add_turn(
        thread_id,
        "assistant",
        "4",
        tool_name="chat",
        model_provider="google",
        model_name="gemini-2.5-flash",
        model_metadata={"usage": {"input_tokens": 3}},
    )

    path = state_dir / "threads" / f"{thread_id}.jsonl"
    header, user_turn, assistant_turn = _read_records(path)

    assert header == {
        "type": "thread",
        "thread_id": thread_id,
        "parent_thread_id": None,
        "tool_name": "chat",
        "created_at": header["created_at"],
    }
    assert "initial_context" not in header

    turn_fields = set(ConversationTurn.model_fields)
    for record in (user_turn, assistant_turn):
        assert record["type"] == "turn"
        assert record["thread_id"] == thread_id
        assert set(record) == turn_fields | {"type", "thread_id"}

    assert user_turn["role"] == "user"
    assert user_turn["content"] == "What is 2+2?"
    assert user_turn["files"] == ["/a.py"]
    assert assistant_turn["content"] == "4"
    assert assistant_turn["model_name"] == "gemini-2.5-flash"
    assert assistant_turn["model_metadata"] == {"usage": {"input_tokens": 3}}


def test_header_records_parent_thread(state_dir):
    parent = conversation_memory.create_thread("chat", {})
    child = conversation_memory.create_thread("codereview", {}, parent_thread_id=parent)

    (header,) = _read_records(state_dir / "threads" / f"{child}.jsonl")
    assert header["parent_thread_id"] == parent
    assert header["tool_name"] == "codereview"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_directory_and_file_modes(state_dir):
    thread_id = conversation_memory.create_thread("chat", {})
    conversation_memory.add_turn(thread_id, "user", "hi")

    threads_dir = state_dir / "threads"
    assert stat.S_IMODE(threads_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((threads_dir / f"{thread_id}.jsonl").stat().st_mode) == 0o600


def test_turn_for_thread_without_file_is_appended(state_dir):
    thread_id = conversation_memory.create_thread("chat", {})
    path = state_dir / "threads" / f"{thread_id}.jsonl"
    path.unlink()  # e.g. the server started mid-thread

    assert conversation_memory.add_turn(thread_id, "user", "hi")
    (record,) = _read_records(path)
    assert record["type"] == "turn"


def test_rejected_turn_is_not_written(state_dir):
    thread_id = conversation_memory.create_thread("chat", {})
    unknown = "00000000-0000-4000-8000-000000000000"

    assert conversation_memory.add_turn(unknown, "user", "hi") is False
    assert not (state_dir / "threads" / f"{unknown}.jsonl").exists()
    assert len(_read_records(state_dir / "threads" / f"{thread_id}.jsonl")) == 1


def test_disabled_writes_nothing(state_dir, monkeypatch):
    monkeypatch.setenv("PAL_TRANSCRIPTS", "false")

    thread_id = conversation_memory.create_thread("chat", {})
    assert conversation_memory.add_turn(thread_id, "user", "hi") is True

    assert not state_dir.exists()


def test_write_failure_is_swallowed_with_warning(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("PAL_STATE_DIR", str(blocker))
    monkeypatch.delenv("PAL_TRANSCRIPTS", raising=False)

    with caplog.at_level(logging.WARNING, logger="utils.transcripts"):
        thread_id = conversation_memory.create_thread("chat", {})
        assert conversation_memory.add_turn(thread_id, "user", "hi") is True

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "utils.transcripts"]
    assert len(warnings) == 2
    assert all(thread_id in r.getMessage() for r in warnings)


def test_unserializable_turn_is_swallowed_with_warning(state_dir, caplog):
    # Storage rejects such a turn before the transcript is reached; this pins
    # that record_turn itself never raises if serialization ever fails.
    turn = ConversationTurn.model_construct(
        role="assistant", content="ok", timestamp="t", model_metadata={"o": object()}
    )

    with caplog.at_level(logging.WARNING, logger="utils.transcripts"):
        transcripts.record_turn("tid", turn)

    assert "tid" in caplog.text
    assert not transcripts.transcript_path("tid").exists()


def _make_transcript(directory, name, age_days):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("{}\n")
    mtime = time.time() - age_days * 86400
    os.utime(path, (mtime, mtime))
    return path


def test_prune_removes_old_files_and_keeps_recent(state_dir):
    threads = state_dir / "threads"
    old = _make_transcript(threads, "old.jsonl", age_days=31)
    recent = _make_transcript(threads, "recent.jsonl", age_days=1)
    unrelated = _make_transcript(threads, "notes.txt", age_days=90)

    assert transcripts.prune_transcripts(retention_days=30) == 1

    assert not old.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_prune_reads_retention_from_env(state_dir, monkeypatch):
    threads = state_dir / "threads"
    old = _make_transcript(threads, "old.jsonl", age_days=5)
    monkeypatch.setenv("PAL_TRANSCRIPT_RETENTION_DAYS", "3")

    assert transcripts.prune_transcripts() == 1
    assert not old.exists()


@pytest.mark.parametrize("days", ["0", "-1"])
def test_prune_disabled_for_zero_or_negative_retention(state_dir, monkeypatch, days):
    old = _make_transcript(state_dir / "threads", "old.jsonl", age_days=400)
    monkeypatch.setenv("PAL_TRANSCRIPT_RETENTION_DAYS", days)

    assert transcripts.prune_transcripts() == 0
    assert old.exists()


def test_retention_defaults_to_30_days(state_dir, monkeypatch):
    monkeypatch.delenv("PAL_TRANSCRIPT_RETENTION_DAYS", raising=False)
    assert transcripts.get_retention_days() == 30


def test_invalid_retention_falls_back_to_default(state_dir, monkeypatch, caplog):
    monkeypatch.setenv("PAL_TRANSCRIPT_RETENTION_DAYS", "forever")
    with caplog.at_level(logging.WARNING, logger="utils.transcripts"):
        assert transcripts.get_retention_days() == 30
    assert "PAL_TRANSCRIPT_RETENTION_DAYS" in caplog.text


def test_prune_without_directory_is_a_no_op(state_dir):
    assert transcripts.prune_transcripts(retention_days=30) == 0
    assert not state_dir.exists()


def test_suite_does_not_write_to_real_state_dir():
    """The autouse conftest fixture points PAL_STATE_DIR away from the user's home."""
    from pathlib import Path

    assert transcripts.get_transcripts_dir().parent != Path.home() / ".local" / "state" / "pal-mcp-server"
    assert os.environ.get("PAL_STATE_DIR")
