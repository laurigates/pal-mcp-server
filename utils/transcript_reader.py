"""Read per-thread JSONL transcripts and render them as markdown.

The write side lives in :mod:`utils.transcripts`; this module only reads. Parsing
and rendering are pure functions so the MCP resource handlers in ``server.py``
stay thin: they validate the URI, load files, and return the rendered text.

Record format: see the module docstring of :mod:`utils.transcripts` and
``docs/logging.md`` ("Conversation Transcripts"). Malformed lines are skipped
with a DEBUG log rather than failing the whole read.
"""

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.transcripts import get_transcripts_dir, transcripts_enabled

logger = logging.getLogger(__name__)

THREADS_INDEX_URI = "pal://threads"
THREAD_URI_PREFIX = f"{THREADS_INDEX_URI}/"
THREAD_URI_TEMPLATE = f"{THREAD_URI_PREFIX}{{thread_id}}"
DEFAULT_LIMIT = 50
_SUFFIX = ".jsonl"
_SNIPPET_CHARS = 60


@dataclass(frozen=True)
class ThreadTranscript:
    """One parsed transcript file."""

    thread_id: str
    tool_name: str | None
    parent_thread_id: str | None
    created_at: str | None
    has_header: bool
    turns: tuple[dict[str, Any], ...]

    @property
    def models(self) -> tuple[str, ...]:
        """Distinct ``model (provider)`` labels of the assistant turns, in first-seen order."""
        labels: list[str] = []
        for turn in self.turns:
            label = _model_label(turn)
            if label and label not in labels:
                labels.append(label)
        return tuple(labels)

    @property
    def last_update(self) -> str | None:
        """Timestamp of the last turn, else the thread's creation time."""
        for turn in reversed(self.turns):
            if turn.get("timestamp"):
                return str(turn["timestamp"])
        return self.created_at

    @property
    def first_prompt(self) -> str | None:
        """Content of the first user turn, if any."""
        for turn in self.turns:
            if turn.get("role") == "user" and turn.get("content"):
                return str(turn["content"])
        return None


def readable_transcripts_dir() -> Path | None:
    """Return the transcripts directory to read, or ``None`` when ``PAL_TRANSCRIPTS`` disables them."""
    return get_transcripts_dir() if transcripts_enabled() else None


def is_valid_thread_id(thread_id: str) -> bool:
    """Return whether ``thread_id`` is a canonical lowercase UUID string (the form PAL writes)."""
    try:
        return str(uuid.UUID(thread_id)) == thread_id
    except (ValueError, TypeError, AttributeError):
        return False


def thread_uri(thread_id: str) -> str:
    """Return the resource URI of one thread."""
    return f"{THREAD_URI_PREFIX}{thread_id}"


def parse_transcript(thread_id: str, lines: Iterable[str]) -> ThreadTranscript:
    """Parse JSONL ``lines`` of one transcript; malformed or unknown lines are skipped."""
    header: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.debug(f"[TRANSCRIPT] Skipping malformed line {number} of thread {thread_id}: {exc}")
            continue
        record_type = record.get("type") if isinstance(record, dict) else None
        if record_type == "thread" and header is None:
            header = record
        elif record_type == "turn":
            turns.append(record)
        else:
            logger.debug(f"[TRANSCRIPT] Skipping unrecognised line {number} of thread {thread_id}")

    tool_name = (header or {}).get("tool_name") or next((t.get("tool_name") for t in turns if t.get("tool_name")), None)
    return ThreadTranscript(
        thread_id=thread_id,
        tool_name=tool_name,
        parent_thread_id=(header or {}).get("parent_thread_id"),
        created_at=(header or {}).get("created_at"),
        has_header=header is not None,
        turns=tuple(turns),
    )


def load_transcript(path: Path) -> ThreadTranscript:
    """Read and parse one transcript file. Raises ``OSError`` if it cannot be read."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        return parse_transcript(path.stem, handle)


def recent_transcript_paths(directory: Path, limit: int = DEFAULT_LIMIT) -> list[Path]:
    """Return up to ``limit`` transcript files named by a valid thread id, newest mtime first.

    A missing or unreadable directory yields an empty list.
    """
    entries: list[tuple[float, Path]] = []
    try:
        candidates = list(directory.glob(f"*{_SUFFIX}"))
    except OSError:
        return []
    for path in candidates:
        if not is_valid_thread_id(path.stem):
            continue
        try:
            if path.is_file():
                entries.append((path.stat().st_mtime, path))
        except OSError:
            continue
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return [path for _, path in entries[: max(limit, 0)]]


def load_recent_transcripts(directory: Path, limit: int = DEFAULT_LIMIT) -> list[ThreadTranscript]:
    """Load the ``limit`` most recently modified transcripts; unreadable files are skipped."""
    transcripts: list[ThreadTranscript] = []
    for path in recent_transcript_paths(directory, limit):
        try:
            transcripts.append(load_transcript(path))
        except OSError as exc:
            logger.debug(f"[TRANSCRIPT] Skipping unreadable transcript {path}: {type(exc).__name__}: {exc}")
    return transcripts


def describe_thread(transcript: ThreadTranscript) -> tuple[str, str]:
    """Return ``(name, description)`` for a thread resource, as shown in a client's ``@`` picker."""
    tool = transcript.tool_name or "thread"
    snippet = _snippet(transcript.first_prompt) if transcript.first_prompt else "(no user prompt)"
    models = ", ".join(transcript.models)
    name = f"{tool}: {snippet}" + (f" ({models})" if models else "")
    turns = len(transcript.turns)
    description = (
        f"PAL {tool} thread {transcript.thread_id}, {turns} turn{'s' if turns != 1 else ''}"
        + (f", models: {models}" if models else "")
        + (f", last update {transcript.last_update}" if transcript.last_update else "")
    )
    return name, description


def render_index(transcripts: Iterable[ThreadTranscript]) -> str:
    """Render the thread index as markdown, in the order given (callers pass newest first)."""
    rows = list(transcripts)
    lines = ["# PAL conversation threads", ""]
    if not rows:
        lines.append(
            "No conversation transcripts found. PAL writes them to `<state dir>/threads/` "
            "unless `PAL_TRANSCRIPTS=false`."
        )
        return "\n".join(lines) + "\n"

    lines += [
        f"{len(rows)} most recent thread{'s' if len(rows) != 1 else ''}, newest first. "
        f"Open one with `@pal:{THREAD_URI_PREFIX}<thread_id>`.",
        "",
        "| Thread | Tool | Models | Turns | Last update |",
        "|---|---|---|---|---|",
    ]
    for t in rows:
        cells = [
            f"`{t.thread_id}`",
            _cell(t.tool_name or "—"),
            _cell(", ".join(t.models) or "—"),
            str(len(t.turns)),
            _cell(t.last_update or "—"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_thread(transcript: ThreadTranscript) -> str:
    """Render one thread as markdown: a header block, then each turn."""
    t = transcript
    lines = [f"# PAL thread `{t.thread_id}`", ""]
    lines.append(f"- **Tool:** {t.tool_name or '—'}")
    if t.created_at:
        lines.append(f"- **Created:** {t.created_at}")
    if t.parent_thread_id:
        lines.append(f"- **Parent thread:** `{t.parent_thread_id}`")
    lines.append(f"- **Models:** {', '.join(t.models) or '—'}")
    lines.append(f"- **Turns:** {len(t.turns)}")
    if not t.has_header:
        lines += ["", "> This transcript starts mid-thread: the server was started after the thread was created."]

    for number, turn in enumerate(t.turns, start=1):
        heading = [f"Turn {number}", str(turn.get("role") or "unknown")]
        label = _model_label(turn)
        if label:
            heading.append(label)
        if turn.get("timestamp"):
            heading.append(str(turn["timestamp"]))
        lines += ["", "---", "", f"## {' · '.join(heading)}", ""]
        if turn.get("tool_name"):
            lines.append(f"- **Tool:** {turn['tool_name']}")
        if turn.get("files"):
            lines.append(f"- **Files:** {', '.join(f'`{f}`' for f in turn['files'])}")
        if turn.get("images"):
            lines.append(f"- **Images:** {', '.join(f'`{i}`' for i in turn['images'])}")
        if turn.get("tool_name") or turn.get("files") or turn.get("images"):
            lines.append("")
        lines.append(str(turn.get("content") or "").rstrip() or "_(empty)_")
    return "\n".join(lines) + "\n"


def _model_label(turn: dict[str, Any]) -> str | None:
    model = turn.get("model_name")
    if not model:
        return None
    provider = turn.get("model_provider")
    return f"{model} ({provider})" if provider else str(model)


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    cut = collapsed[:_SNIPPET_CHARS].rsplit(" ", 1)[0] or collapsed[:_SNIPPET_CHARS]
    return cut + "…"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
