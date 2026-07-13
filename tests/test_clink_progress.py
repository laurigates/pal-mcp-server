"""Progress-reporting wiring for the clink CLI runner.

`utils.progress` is exhaustively unit-tested elsewhere; these tests pin the two
clink-specific hooks that those tests cannot see:

- the exit-status ``update`` fired after the subprocess returns, and
- that a timeout raises through the heartbeat context so its ticker is cancelled
  rather than leaked.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.gemini import GeminiAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from utils.progress import ProgressReporter, set_progress_reporter


def _make_agent(timeout_seconds: int = 30):
    prompt_path = Path("systemprompts/clink/gemini_default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="gemini",
        executable=["gemini"],
        internal_args=[],
        config_args=[],
        env={},
        timeout_seconds=timeout_seconds,
        parser="gemini_json",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return GeminiAgent(client), role


class DummyProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, _input=None):
        return self._stdout, self._stderr


class HangingProcess:
    """Never returns from the first communicate() until it is killed."""

    def __init__(self):
        self.returncode = None
        self.killed = False

    async def communicate(self, _input=None):
        if self.killed:
            return b"", b""
        await asyncio.Event().wait()  # block until cancelled

    def kill(self):
        self.killed = True


def _patch_subprocess(monkeypatch, process):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.mark.asyncio
async def test_reports_exit_status_after_subprocess_returns(monkeypatch):
    agent, role = _make_agent()
    session = AsyncMock()
    set_progress_reporter(ProgressReporter(session=session, progress_token=1))
    _patch_subprocess(monkeypatch, DummyProcess(stdout=b'{"response": "hi there", "stats": {}}'))

    await agent.run(role=role, prompt="do something", files=[], images=[])

    messages = [call.kwargs["message"] for call in session.send_progress_notification.await_args_list]
    assert any(m.startswith("clink · gemini CLI") and "exited 0 in" in m for m in messages), messages


@pytest.mark.asyncio
async def test_timeout_raises_and_heartbeat_ticker_is_cancelled(monkeypatch):
    monkeypatch.setenv("PAL_PROGRESS_HEARTBEAT_SECONDS", "0.01")
    agent, role = _make_agent(timeout_seconds=0)  # wait_for(timeout=0) fires immediately
    session = AsyncMock()
    set_progress_reporter(ProgressReporter(session=session, progress_token=1))
    _patch_subprocess(monkeypatch, HangingProcess())

    with pytest.raises(CLIAgentError, match="timed out"):
        await agent.run(role=role, prompt="do something", files=[], images=[])

    # The heartbeat ticker must not outlive the raise: no further notifications.
    sent = len(session.send_progress_notification.await_args_list)
    await asyncio.sleep(0.05)
    assert len(session.send_progress_notification.await_args_list) == sent
