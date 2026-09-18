"""Tests for the simulator harness's MCP server session (issue #132).

Conversation threads live in a process-local in-memory singleton
(``utils/storage_backend.py``), so a ``continuation_id`` minted by one tool call
is only resolvable by a later call that reaches the *same* server process. The
simulator used to spawn a server per tool call, which made every cross-call
continuation scenario unresolvable by construction -- the scenarios reported a
harness limitation as a server failure.

These tests pin the process-identity property that fix depends on. They use the
``version`` tool, which needs no provider credentials, so they run in the normal
unit suite rather than under the integration marker.
"""

import json
import logging

import pytest

from simulator_tests.base_test import BaseSimulatorTest, MCPServerSession


class _Harness(BaseSimulatorTest):
    """Minimal concrete subclass; the base class is abstract on these two."""

    @property
    def test_name(self) -> str:
        return "server_session_probe"

    def run_test(self) -> bool:
        raise NotImplementedError


@pytest.fixture
def harness():
    return _Harness(verbose=False)


class TestMCPServerSession:
    def test_session_reuses_one_process_across_calls(self, harness):
        """The whole point: call two and call one must be the same process."""
        with harness.server_session(timeout=120) as session:
            session.ensure_started()
            first_pid = session._proc.pid

            assert session.call_tool("version", {}) is not None
            assert session.call_tool("version", {}) is not None

            assert session.is_alive
            assert session._proc.pid == first_pid

    def test_request_ids_advance_across_calls(self, harness):
        """Each call must read its own reply, not the previous call's."""
        with harness.server_session(timeout=120) as session:
            session.ensure_started()

            first_id = session.next_response_id()
            first = session.call_tool("version", {})
            second_id = session.next_response_id()
            second = session.call_tool("version", {})

            assert second_id == first_id + 1
            assert _response_id_present(first, first_id)
            assert _response_id_present(second, second_id)

    def test_calls_outside_a_session_use_separate_processes(self, harness):
        """The old behaviour is still what a non-continuation call gets."""
        pids = []
        for _ in range(2):
            with MCPServerSession(harness.python_path, logging.getLogger(__name__), timeout=120) as one_shot:
                one_shot.call_tool("version", {})
                pids.append(one_shot._proc.pid)

        assert pids[0] != pids[1]

    def test_session_closes_cleanly(self, harness):
        session = MCPServerSession(harness.python_path, logging.getLogger(__name__), timeout=120)
        session.start()
        assert session.is_alive
        session.close()
        assert not session.is_alive

    def test_unstarted_session_costs_nothing(self, harness):
        """server_session() wraps every test, including in-process-only ones."""
        with harness.server_session(timeout=120) as session:
            assert session._proc is None
            assert not session.is_alive


class TestClientDefaults:
    def test_chat_gets_a_working_directory(self, harness):
        """chat requires working_directory_absolute_path; no scenario passed it."""
        params = harness.apply_client_defaults("chat", {"prompt": "hi"})

        assert "working_directory_absolute_path" in params

    def test_explicit_working_directory_is_not_overridden(self, harness):
        params = harness.apply_client_defaults("chat", {"prompt": "hi", "working_directory_absolute_path": "/tmp"})

        assert params["working_directory_absolute_path"] == "/tmp"

    def test_other_tools_are_untouched(self, harness):
        """Deliberately narrow, so a new required field still fails loudly."""
        original = {"prompt": "hi"}
        params = harness.apply_client_defaults("thinkdeep", original)

        assert params == original

    def test_caller_dict_is_not_mutated(self, harness):
        original = {"prompt": "hi"}
        harness.apply_client_defaults("chat", original)

        assert original == {"prompt": "hi"}


def _response_id_present(raw_lines: str, expected_id: int) -> bool:
    for line in raw_lines.splitlines():
        if line.startswith("{") and json.loads(line).get("id") == expected_id:
            return True
    return False
