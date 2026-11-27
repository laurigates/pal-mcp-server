"""Normalised token-usage keys shared by the provider usage parsers.

Every parser writes the same keys onto ``ModelResponse.usage`` so the status
line, transcripts and ``just cache-report`` can read one shape regardless of
which wire format the provider spoke (issue #193, docs/prompt-caching.md).

``CACHED_INPUT_TOKENS`` is the part of the input served from the provider's
prompt cache; ``CACHE_WRITE_INPUT_TOKENS`` the part written to it. A parser
writes a key only when the provider reported the field (Gemini excepted, whose
SDK omits it on a miss), so a report can measure field coverage from absence.
"""

from __future__ import annotations

from typing import Any

CACHED_INPUT_TOKENS = "cached_input_tokens"
CACHE_WRITE_INPUT_TOKENS = "cache_write_input_tokens"


def int_field(obj: Any, name: str) -> int | None:
    """Return ``obj.name`` when it is a real int, else None.

    Provider SDK objects carry optional fields as None, and tests build responses
    from bare ``Mock()`` objects whose attributes are auto-created mocks; neither
    may be recorded as a token count. ``bool`` is an ``int`` subclass and is
    excluded for the same reason.
    """
    if obj is None:
        return None
    try:
        value = getattr(obj, name, None)
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
