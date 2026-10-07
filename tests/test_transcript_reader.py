"""Unit tests for utils.transcript_reader (parse + render of per-thread JSONL transcripts)."""

import json
import logging
import os
import uuid

import pytest

from utils import transcript_reader as reader

THREAD_ID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"


def _header(thread_id=THREAD_ID, tool="chat"):
    return {
        "type": "thread",
        "thread_id": thread_id,
        "parent_thread_id": None,
        "tool_name": tool,
        "created_at": "2026-10-07T10:00:00+00:00",
    }


def _turn(role, content, ts, model=None, provider=None, files=None, tool="chat", thread_id=THREAD_ID):
    return {
        "type": "turn",
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "timestamp": ts,
        "files": files,
        "images": None,
        "tool_name": tool,
        "model_provider": provider,
        "model_name": model,
        "model_metadata": None,
    }


def _lines(*records):
    return [json.dumps(r) + "\n" for r in records]


def test_header_first_file():
    t = reader.parse_transcript(
        THREAD_ID,
        _lines(
            _header(),
            _turn("user", "Review the auth flow", "2026-10-07T10:00:01+00:00", files=["/src/auth.py"]),
            _turn("assistant", "Looks fine.", "2026-10-07T10:00:05+00:00", model="gpt-5", provider="openai"),
        ),
    )
    assert t.has_header
    assert t.tool_name == "chat"
    assert t.created_at == "2026-10-07T10:00:00+00:00"
    assert len(t.turns) == 2
    assert t.models == ("gpt-5 (openai)",)
    assert t.last_update == "2026-10-07T10:00:05+00:00"
    assert t.first_prompt == "Review the auth flow"


def test_turn_first_file_takes_tool_from_turn_and_notes_mid_thread():
    t = reader.parse_transcript(
        THREAD_ID,
        _lines(_turn("assistant", "Second reply", "2026-10-07T11:00:00+00:00", model="flash", tool="thinkdeep")),
    )
    assert not t.has_header
    assert t.tool_name == "thinkdeep"
    assert t.created_at is None
    assert t.last_update == "2026-10-07T11:00:00+00:00"
    assert "starts mid-thread" in reader.render_thread(t)


def test_malformed_line_skipped_with_debug_log(caplog):
    lines = _lines(_header()) + ["{not json\n", "[1, 2]\n", "\n"] + _lines(_turn("user", "hi", "t1"))
    with caplog.at_level(logging.DEBUG, logger="utils.transcript_reader"):
        t = reader.parse_transcript(THREAD_ID, lines)
    assert len(t.turns) == 1
    assert t.has_header
    assert any("malformed line 2" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "value",
    [
        "../etc/passwd",
        "..%2F..%2Fsecret",
        "not-a-uuid",
        "",
        THREAD_ID.upper(),
        THREAD_ID.replace("-", ""),
        f"{THREAD_ID}/../x",
        f"{{{THREAD_ID}}}",
    ],
)
def test_invalid_thread_id_rejected(value):
    assert not reader.is_valid_thread_id(value)


def test_valid_thread_id_accepted():
    assert reader.is_valid_thread_id(THREAD_ID)
    assert reader.is_valid_thread_id(str(uuid.uuid4()))


def test_recent_paths_capped_newest_first_and_skip_non_uuid_names(tmp_path):
    ids = [str(uuid.UUID(int=i + 1)) for i in range(55)]
    for i, tid in enumerate(ids):
        path = tmp_path / f"{tid}.jsonl"
        path.write_text(json.dumps(_header(tid)) + "\n")
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
    stray = tmp_path / "notes.jsonl"
    stray.write_text("{}\n")
    os.utime(stray, (9_000_000, 9_000_000))

    paths = reader.recent_transcript_paths(tmp_path)
    assert len(paths) == reader.DEFAULT_LIMIT == 50
    assert [p.stem for p in paths] == list(reversed(ids))[:50]


def test_recent_paths_missing_directory_is_empty(tmp_path):
    assert reader.recent_transcript_paths(tmp_path / "absent") == []
    assert reader.load_recent_transcripts(tmp_path / "absent") == []


def test_render_index_columns_and_empty_state():
    t = reader.parse_transcript(
        THREAD_ID,
        _lines(
            _header(),
            _turn("user", "q", "2026-10-07T10:00:01+00:00"),
            _turn("assistant", "a", "2026-10-07T10:00:02+00:00", model="gemini-2.5-pro", provider="google"),
        ),
    )
    index = reader.render_index([t])
    assert "| Thread | Tool | Models | Turns | Last update |" in index
    assert f"| `{THREAD_ID}` | chat | gemini-2.5-pro (google) | 2 | 2026-10-07T10:00:02+00:00 |" in index
    assert "No conversation transcripts found" in reader.render_index([])


def test_render_thread_shows_role_model_timestamp_files_content():
    t = reader.parse_transcript(
        THREAD_ID,
        _lines(
            _header(),
            _turn("user", "Review auth", "2026-10-07T10:00:01+00:00", files=["/src/auth.py"]),
            _turn("assistant", "Looks fine.", "2026-10-07T10:00:05+00:00", model="gpt-5", provider="openai"),
        ),
    )
    md = reader.render_thread(t)
    assert md.startswith(f"# PAL thread `{THREAD_ID}`")
    assert "## Turn 1 · user · 2026-10-07T10:00:01+00:00" in md
    assert "## Turn 2 · assistant · gpt-5 (openai) · 2026-10-07T10:00:05+00:00" in md
    assert "- **Files:** `/src/auth.py`" in md
    assert "Looks fine." in md


def test_describe_thread_names_tool_prompt_snippet_and_model():
    long_prompt = "Please review the authentication module for race conditions and token reuse " * 3
    t = reader.parse_transcript(
        THREAD_ID,
        _lines(
            _header(tool="codereview"),
            _turn("user", long_prompt, "t1"),
            _turn("assistant", "ok", "t2", model="o3", provider="openai"),
        ),
    )
    name, description = reader.describe_thread(t)
    assert name.startswith("codereview: Please review the authentication")
    assert "…" in name
    assert name.endswith("(o3 (openai))")
    assert THREAD_ID in description
    assert "2 turns" in description
