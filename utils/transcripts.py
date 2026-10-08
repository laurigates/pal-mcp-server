"""Per-thread JSONL transcripts of PAL conversation turns.

Each conversation thread is appended to ``<state_dir>/threads/<thread_id>.jsonl``
(see :func:`utils.state_dir.get_state_dir`), so a human can read or ``tail -f``
what delegated models were asked and answered. The in-memory thread store is
unchanged; this is a write-only side channel.

Record format (one JSON object per line, UTF-8):

* ``{"type": "thread", "thread_id", "parent_thread_id", "tool_name", "created_at"}``
  -- written once by ``create_thread()``. ``initial_context`` is deliberately
  omitted to keep the header small.
* ``{"type": "turn", "thread_id", ...}`` -- one per ``add_turn()``, carrying every
  ``ConversationTurn`` field (``role``, ``content``, ``timestamp``, ``files``,
  ``images``, ``tool_name``, ``model_provider``, ``model_name``,
  ``model_metadata``). Written only after the turn was saved to storage.

A file may start with a turn record when the server started mid-thread.

Configuration: ``PAL_TRANSCRIPTS=false`` disables writing (on by default).
``PAL_TRANSCRIPT_RETENTION_DAYS`` (default 30; ``0`` or negative never prunes)
controls the prune by file mtime, which runs once at server startup only. Every
write sets the directory to ``0700`` and the file to ``0600``, including ones
that already existed. A write or prune failure is logged at WARNING and never
propagates.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from utils.env import get_env, get_env_bool
from utils.state_dir import get_state_dir

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 30
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_SUFFIX = ".jsonl"


def transcripts_enabled() -> bool:
    """Return whether transcripts are written (``PAL_TRANSCRIPTS``, default true)."""
    return get_env_bool("PAL_TRANSCRIPTS", default=True)


def get_transcripts_dir() -> Path:
    """Return the directory holding per-thread transcript files."""
    return get_state_dir() / "threads"


def transcript_path(thread_id: str) -> Path:
    """Return the transcript file path for ``thread_id``."""
    return get_transcripts_dir() / f"{thread_id}{_SUFFIX}"


def append_record(thread_id: str, record: dict[str, Any]) -> None:
    """Append one JSON record to the thread's transcript; never raises."""
    if not transcripts_enabled():
        return
    try:
        path = transcript_path(thread_id)
        path.parent.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
        # mkdir/os.open apply the mode only on creation; tighten pre-existing ones too.
        path.parent.chmod(_DIR_MODE)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _FILE_MODE)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), _FILE_MODE)
            handle.write(line)
            handle.flush()
    except Exception as exc:
        logger.warning(f"[TRANSCRIPT] Could not write transcript for thread {thread_id}: {type(exc).__name__}: {exc}")


def record_thread_created(thread_id: str, parent_thread_id: str | None, tool_name: str, created_at: str) -> None:
    """Write the thread header record."""
    append_record(
        thread_id,
        {
            "type": "thread",
            "thread_id": thread_id,
            "parent_thread_id": parent_thread_id,
            "tool_name": tool_name,
            "created_at": created_at,
        },
    )


def record_turn(thread_id: str, turn: BaseModel) -> None:
    """Write a turn record carrying every field of a ``ConversationTurn``; never raises."""
    if not transcripts_enabled():
        return
    try:
        turn_fields = turn.model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"[TRANSCRIPT] Could not serialize turn for thread {thread_id}: {type(exc).__name__}: {exc}")
        return
    append_record(thread_id, {"type": "turn", "thread_id": thread_id, **turn_fields})


def get_retention_days() -> int:
    """Return ``PAL_TRANSCRIPT_RETENTION_DAYS``, falling back to the default when unparseable."""
    raw = (get_env("PAL_TRANSCRIPT_RETENTION_DAYS") or "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Invalid PAL_TRANSCRIPT_RETENTION_DAYS value ({raw!r}), using default of {DEFAULT_RETENTION_DAYS} days"
        )
        return DEFAULT_RETENTION_DAYS


def prune_transcripts(retention_days: int | None = None) -> int:
    """Delete transcript files whose mtime is older than the retention window.

    Returns the number of files removed. ``retention_days`` of 0 or less keeps
    everything. Failures are logged at WARNING and never raised.
    """
    try:
        return _prune(get_retention_days() if retention_days is None else retention_days)
    except Exception as exc:
        logger.warning(f"[TRANSCRIPT] Could not prune transcripts: {type(exc).__name__}: {exc}")
        return 0


def _prune(days: int) -> int:
    if days <= 0:
        return 0

    directory = get_transcripts_dir()
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        candidates = list(directory.glob(f"*{_SUFFIX}"))
    except OSError as exc:
        logger.warning(f"[TRANSCRIPT] Could not list {directory} for pruning: {type(exc).__name__}: {exc}")
        return 0

    for path in candidates:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning(f"[TRANSCRIPT] Could not prune {path}: {type(exc).__name__}: {exc}")

    if removed:
        logger.info(f"[TRANSCRIPT] Pruned {removed} transcript(s) older than {days} days from {directory}")
    return removed
