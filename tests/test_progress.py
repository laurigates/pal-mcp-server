"""Tests for MCP progress reporting.

Two layers:

- Unit tests over `utils.progress`, including the invariant that matters most:
  a progress failure must never surface to the tool call it describes.
- A live stdio probe that drives the real server with a real MCP client which
  supplies a `progressToken`, proving the notifications actually reach a client
  in the right wire format. Unit tests with a mocked session cannot show that.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.progress import (
    ProgressReporter,
    format_duration,
    format_tokens,
    get_progress_reporter,
    reporter_from_request_context,
    set_progress_reporter,
    summarize_usage,
    usage_metadata,
)


class TestFormatting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0s"), (9.4, "9s"), (59, "59s"), (60, "1m00s"), (64, "1m04s"), (3600, "1h00m"), (3725, "1h02m")],
    )
    def test_format_duration(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_format_duration_clamps_negative(self):
        assert format_duration(-5) == "0s"

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(0, "0"), (812, "812"), (12400, "12.4k"), (1_200_000, "1.2M")],
    )
    def test_format_tokens(self, count, expected):
        assert format_tokens(count) == expected

    def test_summarize_usage(self):
        assert summarize_usage({"input_tokens": 12400, "output_tokens": 3100}) == "12.4k in → 3.1k out"

    def test_summarize_usage_falls_back_to_total(self):
        assert summarize_usage({"total_tokens": 900}) == "900 tokens"

    def test_summarize_usage_without_data(self):
        assert summarize_usage({}) == "done"
        assert summarize_usage(None) == "done"

    @pytest.mark.parametrize(
        "usage",
        [
            MagicMock(),  # a mocked ModelResponse hands back a Mock, not a dict
            {"input_tokens": MagicMock()},
            {"input_tokens": "1200"},
            {"input_tokens": None},
            {"input_tokens": -5},
            "not-a-dict",
        ],
    )
    def test_malformed_usage_degrades_instead_of_raising(self, usage):
        """A status line must never be able to raise into the tool call it describes."""
        assert summarize_usage(usage) == "done"
        assert usage_metadata(usage) == {}


class TestUsageMetadata:
    def test_includes_token_counts_and_duration(self):
        metadata = usage_metadata({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, 4.27)
        assert metadata == {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "duration_seconds": 4.3,
        }

    def test_omits_absent_fields(self):
        assert usage_metadata(None) == {}
        assert usage_metadata({"input_tokens": 0}) == {}


class TestProgressReporterDisabled:
    """No progressToken means no channel; every call must be a silent no-op."""

    @pytest.mark.asyncio
    async def test_update_without_session_is_noop(self):
        reporter = ProgressReporter()
        assert reporter.enabled is False
        await reporter.update("anything")  # must not raise

    @pytest.mark.asyncio
    async def test_session_without_token_sends_nothing(self):
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token=None)
        await reporter.update("anything")
        session.send_progress_notification.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_without_token_still_runs_body(self):
        reporter = ProgressReporter()
        ran = False
        async with reporter.heartbeat("working"):
            ran = True
        assert ran


class TestProgressReporterUpdate:
    @pytest.mark.asyncio
    async def test_sends_notification_with_token_and_request_id(self):
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token="tok-1", request_id=7)

        await reporter.update("chat · gpt-5 · thinking")

        session.send_progress_notification.assert_awaited_once_with(
            progress_token="tok-1",
            progress=1.0,
            message="chat · gpt-5 · thinking",
            related_request_id=7,
        )

    @pytest.mark.asyncio
    async def test_progress_increases_monotonically(self):
        """The MCP spec requires `progress` to increase on every notification."""
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token=1)

        for _ in range(3):
            await reporter.update("tick")

        values = [call.kwargs["progress"] for call in session.send_progress_notification.await_args_list]
        assert values == [1.0, 2.0, 3.0]
        assert values == sorted(set(values))

    @pytest.mark.asyncio
    async def test_collapses_whitespace_and_truncates_to_client_limit(self):
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token=1)

        await reporter.update("a  b\n\tc " + "x" * 400)

        message = session.send_progress_notification.await_args.kwargs["message"]
        assert message.startswith("a b c ")
        assert len(message) == 200
        assert message.endswith("…")

    @pytest.mark.asyncio
    async def test_notification_failure_never_reaches_the_caller(self):
        """A courtesy notification must not be able to fail the tool call."""
        session = AsyncMock()
        session.send_progress_notification.side_effect = RuntimeError("transport closed")
        reporter = ProgressReporter(session=session, progress_token=1)

        await reporter.update("still fine")  # must swallow


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_ticks_with_a_running_clock_and_stops_on_exit(self, monkeypatch):
        monkeypatch.setenv("PAL_PROGRESS_HEARTBEAT_SECONDS", "0.01")
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token=1)

        async with reporter.heartbeat("clink · gemini CLI · running"):
            await asyncio.sleep(0.08)

        messages = [call.kwargs["message"] for call in session.send_progress_notification.await_args_list]
        assert len(messages) > 2, "heartbeat should re-report while the body runs"
        assert messages[0] == "clink · gemini CLI · running"
        assert all(m.startswith("clink · gemini CLI · running") for m in messages)
        # Later ticks carry an elapsed clock, which is what distinguishes slow from stalled.
        assert any("·" in m and m.rstrip().endswith("s") for m in messages[1:])

        # Once the block exits the ticker must be gone, not leaked.
        sent_after_exit = len(session.send_progress_notification.await_args_list)
        await asyncio.sleep(0.05)
        assert len(session.send_progress_notification.await_args_list) == sent_after_exit

    @pytest.mark.asyncio
    async def test_body_exception_propagates_and_ticker_is_cancelled(self, monkeypatch):
        monkeypatch.setenv("PAL_PROGRESS_HEARTBEAT_SECONDS", "0.01")
        session = AsyncMock()
        reporter = ProgressReporter(session=session, progress_token=1)

        with pytest.raises(ValueError, match="boom"):
            async with reporter.heartbeat("working"):
                raise ValueError("boom")

        sent = len(session.send_progress_notification.await_args_list)
        await asyncio.sleep(0.05)
        assert len(session.send_progress_notification.await_args_list) == sent


class TestReporterFromRequestContext:
    def test_missing_context_is_inert(self):
        assert reporter_from_request_context(None).enabled is False

    def test_context_without_progress_token_is_inert(self):
        class Ctx:
            meta = None
            session = object()
            request_id = 1

        assert reporter_from_request_context(Ctx()).enabled is False

    def test_context_with_progress_token_is_enabled(self):
        class Meta:
            progressToken = "tok"

        class Ctx:
            meta = Meta()
            session = object()
            request_id = 3

        assert reporter_from_request_context(Ctx()).enabled is True


class TestContextVar:
    def test_default_reporter_is_inert(self):
        assert get_progress_reporter().enabled is False

    @pytest.mark.asyncio
    async def test_reporter_reaches_code_that_never_received_it(self):
        """The point of the contextvar: deep code reports without being handed a reporter."""
        session = AsyncMock()
        set_progress_reporter(ProgressReporter(session=session, progress_token=1))

        async def deep_in_a_provider():
            await get_progress_reporter().update("from the depths")

        await asyncio.create_task(deep_in_a_provider())

        session.send_progress_notification.assert_awaited_once()
        assert session.send_progress_notification.await_args.kwargs["message"] == "from the depths"

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_share_a_reporter(self):
        """Clients batch calls ("Calling pal 2 times"), and the server handles each in
        its own task. Each call's progress must go to its own token, not the other's."""
        session_a, session_b = AsyncMock(), AsyncMock()

        async def one_call(session, token, message):
            # Mirrors the server: a task per request, reporter published inside it.
            set_progress_reporter(ProgressReporter(session=session, progress_token=token))
            await asyncio.sleep(0)  # let the sibling interleave
            await get_progress_reporter().update(message)

        await asyncio.gather(
            asyncio.create_task(one_call(session_a, "tok-a", "call A")),
            asyncio.create_task(one_call(session_b, "tok-b", "call B")),
        )

        for session, token, message in ((session_a, "tok-a", "call A"), (session_b, "tok-b", "call B")):
            session.send_progress_notification.assert_awaited_once()
            kwargs = session.send_progress_notification.await_args.kwargs
            assert kwargs["progress_token"] == token
            assert kwargs["message"] == message


@pytest.mark.asyncio
async def test_live_stdio_client_receives_progress_notifications(tmp_path):
    """Drive the real server over stdio with a real client that opts into progress.

    This is the end-to-end proof: a mocked session can show we *call* the SDK, but
    only a real client round-trip shows the notification is well-formed enough to
    be routed back to the caller's progress handler.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    repo_root = Path(__file__).resolve().parent.parent
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server"],
        cwd=str(repo_root),
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": str(repo_root),
            "LOG_LEVEL": "ERROR",
            # A key is required for the server to boot with a usable provider registry;
            # `version` never calls out, so this is never spent.
            "OPENAI_API_KEY": "sk-test-not-used",
            "DEFAULT_MODEL": "auto",
        },
    )

    received: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        received.append((progress, total, message))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("version", {}, progress_callback=on_progress)

    assert not result.isError, result.content

    assert received, "server sent no progress notifications to a client that supplied a progressToken"
    messages = [m for _, _, m in received]
    assert any(m and m.startswith("version ·") for m in messages), messages

    # Spec: progress must strictly increase so a client can order notifications.
    progresses = [p for p, _, _ in received]
    assert progresses == sorted(progresses)
    assert len(set(progresses)) == len(progresses)
