"""Publish a stable conversation identifier for provider request headers.

Some gateways want each request tagged with the conversation it belongs to -
OpenCode Go requires an ``x-opencode-session`` header ("one stable ID per
conversation") and rejects requests without one. PAL already has that identity:
the ``continuation_id`` threading a multi-turn exchange through
:mod:`utils.conversation_memory`.

Passing it down to a provider through every tool signature would touch every
call site, so it travels the way the progress reporter does - a ContextVar the
server publishes for the in-flight call, read by whoever needs it. The first
call of a conversation has no continuation_id yet (the thread is created while
it runs), so that case falls back to a per-process ID and the header is never
missing.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

# Header values must stay ASCII and free of control characters. continuation_ids
# are UUIDs, but they arrive from the client, so anything that does not look
# like an identifier is not trusted into a header.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

# One MCP stdio session is one server process, so this is the coarsest sensible
# conversation ID: stable for as long as the client is connected.
_PROCESS_SESSION_ID = f"pal-{uuid.uuid4().hex}"

_current_session_id: ContextVar[str | None] = ContextVar("pal_session_id", default=None)


def get_session_id() -> str:
    """Return the ID of the in-flight conversation, or the process-wide one."""
    return _current_session_id.get() or _PROCESS_SESSION_ID


def set_session_id(session_id: object) -> None:
    """Publish the conversation ID for the current call and the tasks it spawns.

    Anything that is not a header-safe identifier - including ``None`` - clears
    the override, leaving :func:`get_session_id` on the process-wide ID.
    """
    if isinstance(session_id, str) and _SAFE_SESSION_ID.match(session_id):
        _current_session_id.set(session_id)
        return
    _current_session_id.set(None)
