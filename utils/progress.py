"""Emit MCP progress notifications so clients can show live tool status.

Long PAL tool calls are a black box to the caller: a single `await` on a
provider or a CLI subprocess can run for minutes, and the client has no way to
tell "thinking" from "wedged". The MCP `notifications/progress` channel is the
only in-flight, non-blocking surface to the user, so tools report through it.

Two things make this worth doing beyond cosmetics:

- The client opts in by sending ``_meta.progressToken`` with the call. Without a
  token there is nowhere to send notifications and everything here no-ops.
- Notifications reset the client's *idle* timeout. A call that emits nothing can
  be aborted for idleness even though it is making progress, so the heartbeat is
  a keepalive as much as a status line.

Progress reporting is strictly best-effort: a failure to notify must never fail
the tool call it is describing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

# Clients render only the most recent notification, so a heartbeat replaces the
# status line rather than appending to it. Ticking roughly every few seconds
# reads as "alive" without being chatty on the wire.
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 5.0

# Claude Code truncates the rendered message at 200 characters. Truncate here so
# the elapsed-time suffix is never the part that gets cut.
MAX_MESSAGE_LENGTH = 200


def _heartbeat_interval() -> float:
    raw = os.getenv("PAL_PROGRESS_HEARTBEAT_SECONDS")
    if not raw:
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    try:
        interval = float(raw)
    except ValueError:
        logger.debug("Ignoring non-numeric PAL_PROGRESS_HEARTBEAT_SECONDS=%r", raw)
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    return interval if interval > 0 else DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def format_duration(seconds: float) -> str:
    """Render an elapsed duration compactly: 42s, 1m04s, 1h02m."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_tokens(count: int) -> str:
    """Render a token count compactly: 812, 12.4k, 1.2M."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def _token_count(usage: Any, key: str) -> int | None:
    """Read one token count, tolerating a `usage` that isn't a well-formed int dict.

    Usage is provider-supplied and only feeds a status line, so a missing key or a
    surprising type must degrade to "no number to show" rather than raise into the
    tool call being described.
    """
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def summarize_usage(usage: dict[str, int] | None) -> str:
    """Render provider token usage for a status line: `12.4k in → 3.1k out`."""
    parts = []
    for key, label in (("input_tokens", "in"), ("output_tokens", "out")):
        count = _token_count(usage, key)
        if count is not None:
            parts.append(f"{format_tokens(count)} {label}")

    if not parts:
        total = _token_count(usage, "total_tokens")
        if total is not None:
            parts.append(f"{format_tokens(total)} tokens")

    return " → ".join(parts) if parts else "done"


def usage_metadata(usage: dict[str, int] | None, duration_seconds: float | None = None) -> dict[str, Any]:
    """Build the telemetry block the *calling agent* sees in the tool result.

    Progress notifications only reach the user's terminal — nothing can reach the
    agent mid-call, so cost and latency have to ride back in the result metadata
    or the agent stays blind to how expensive its own delegation was.
    """
    metadata: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        count = _token_count(usage, key)
        if count is not None:
            metadata[key] = count
    if duration_seconds is not None:
        metadata["duration_seconds"] = round(duration_seconds, 1)
    return metadata


class ProgressReporter:
    """Sends `notifications/progress` for a single in-flight tool call.

    Constructed at the MCP boundary from the request context and published on a
    contextvar, so code deep in a provider or CLI runner can report without the
    reporter being threaded through every signature.

    A reporter with no session or no progress token is inert: every method is a
    no-op. That is the normal state for unit tests and for clients that do not
    opt into progress.
    """

    def __init__(
        self,
        session: Any | None = None,
        progress_token: str | int | None = None,
        request_id: str | int | None = None,
    ) -> None:
        self._session = session
        self._progress_token = progress_token
        self._request_id = request_id
        # The MCP spec requires `progress` to increase on every notification.
        # A plain tick counter satisfies that; the meaning lives in the message.
        self._tick = 0
        self._started_at = time.monotonic()

    @property
    def enabled(self) -> bool:
        return self._session is not None and self._progress_token is not None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at

    async def update(self, message: str) -> None:
        """Report current status. Never raises."""
        session = self._session
        if session is None or self._progress_token is None:
            return

        message = " ".join(message.split())
        if len(message) > MAX_MESSAGE_LENGTH:
            message = message[: MAX_MESSAGE_LENGTH - 1] + "…"

        self._tick += 1
        try:
            await session.send_progress_notification(
                progress_token=self._progress_token,
                progress=float(self._tick),
                message=message,
                related_request_id=self._request_id,
            )
        except Exception:
            # A progress notification is a courtesy. Losing one must not take
            # down the tool call it describes, so swallow and record at DEBUG.
            logger.debug("Failed to send progress notification", exc_info=True)

    @asynccontextmanager
    async def heartbeat(self, message: str) -> AsyncIterator[None]:
        """Re-report `message` with a running clock until the block exits.

        Wrap any await long enough that silence is ambiguous — a provider call, a
        CLI subprocess. The caller sees `chat · gpt-5 · thinking · 1m04s` advance,
        which distinguishes slow from stalled, and the traffic keeps the client's
        idle timeout from firing.
        """
        if not self.enabled:
            yield
            return

        started = time.monotonic()
        interval = _heartbeat_interval()

        async def tick() -> None:
            while True:
                elapsed = time.monotonic() - started
                await self.update(f"{message} · {format_duration(elapsed)}")
                await asyncio.sleep(interval)

        await self.update(message)
        task = asyncio.create_task(tick())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


_NULL_REPORTER = ProgressReporter()

_current_reporter: ContextVar[ProgressReporter] = ContextVar("pal_progress_reporter", default=_NULL_REPORTER)


def get_progress_reporter() -> ProgressReporter:
    """Return the reporter for the in-flight call, or an inert one."""
    return _current_reporter.get()


def set_progress_reporter(reporter: ProgressReporter) -> None:
    """Publish the reporter for the current call and any tasks it spawns."""
    _current_reporter.set(reporter)


def reporter_from_request_context(request_context: Any | None) -> ProgressReporter:
    """Build a reporter from an MCP request context, tolerating its absence.

    The context is missing outside a live request (unit tests, direct tool
    invocation) and carries no `progressToken` when the client did not opt in.
    Both yield an inert reporter rather than an error.
    """
    if request_context is None:
        return ProgressReporter()

    meta = getattr(request_context, "meta", None)
    progress_token = getattr(meta, "progressToken", None) if meta else None
    if progress_token is None:
        return ProgressReporter()

    return ProgressReporter(
        session=getattr(request_context, "session", None),
        progress_token=progress_token,
        request_id=getattr(request_context, "request_id", None),
    )
