#!/usr/bin/env python3
"""
Base Test Class for Communication Simulator Tests

Provides common functionality and utilities for all simulator tests.
"""

import json
import logging
import os
import subprocess
import threading
from collections import deque
from contextlib import contextmanager

from .log_utils import LogUtils


class MCPServerSession:
    """One long-lived ``server.py`` subprocess speaking MCP over stdio.

    Conversation threads live in an in-memory, process-local singleton
    (``utils/storage_backend.py``), so a ``continuation_id`` minted by one tool
    call is only resolvable by a later call that reaches the *same* process.
    This harness used to spawn a server per call, which made every cross-call
    continuation scenario unresolvable by construction rather than by
    regression -- the failure those scenarios reported was the harness's, not
    the server's (issue #132).

    Keeping the process alive across a scenario's calls fixes that without a
    persistent storage backend, and without giving up the stdio boundary that
    makes this suite a wire-level check in the first place.

    The session is single-threaded by design: one request is in flight at a
    time, and each call reads until it sees its own id.
    """

    # The handshake occupies id 1; tool calls start after it.
    _FIRST_TOOL_CALL_ID = 2

    def __init__(self, python_path: str, logger: logging.Logger, timeout: int = 3600):
        self.python_path = python_path
        self.logger = logger
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._next_id = self._FIRST_TOOL_CALL_ID
        # Bounded so a chatty server cannot grow this without limit over a long
        # session; only the tail is ever useful for diagnosing a failure.
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None
        self._dead_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def ensure_started(self) -> None:
        """Start on first use, so a session that is never called costs nothing."""
        if self._proc is None:
            self.start()

    def start(self) -> None:
        """Spawn the server and complete the MCP handshake once."""
        self._proc = subprocess.Popen(
            [self.python_path, "server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # stderr must be drained continuously: over a multi-call session a full
        # pipe buffer would block the server mid-response, which looks exactly
        # like a hang. The one-shot-per-call design never ran long enough to hit
        # this, so the drain thread is new with the persistent session.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        handshake = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "communication-simulator", "version": "1.0.0"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ]
        self._write("\n".join(json.dumps(m, ensure_ascii=False) for m in handshake) + "\n")
        if self._read_until_id(1) is None:
            raise RuntimeError(f"MCP handshake failed: {self._dead_reason or 'no initialize response'}")
        self.logger.debug("MCP server session ready (pid %s)", self._proc.pid)

    def close(self) -> None:
        """Shut the server down, closing stdin first so it exits on EOF."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        if self._stderr_thread:
            self._stderr_thread.join(timeout=5)
        self.logger.debug("MCP server session closed")

    def __enter__(self) -> "MCPServerSession":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- requests ----------------------------------------------------------

    def call_tool(self, tool_name: str, params: dict) -> str | None:
        """Send one ``tools/call`` and return the raw JSON-RPC lines it produced.

        Returns None if the session is dead or the call timed out; the reason is
        logged and left on ``_dead_reason`` for the caller to report.
        """
        if not self.is_alive:
            self.logger.error("MCP server session is not running (%s)", self._dead_reason or "never started")
            return None

        request_id = self._next_id
        self._next_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
        }
        self.logger.debug("Calling MCP tool %s (id %s)", tool_name, request_id)
        self._write(json.dumps(request, ensure_ascii=False) + "\n")
        return self._read_until_id(request_id)

    def next_response_id(self) -> int:
        """The id the next :meth:`call_tool` will use, for response parsing."""
        return self._next_id

    # -- plumbing ----------------------------------------------------------

    def _write(self, payload: str) -> None:
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _read_until_id(self, expected_id: int) -> str | None:
        """Read stdout until the reply to ``expected_id`` arrives.

        ``readline()`` has no timeout of its own, so killing the process is what
        unblocks it -- the watchdog owns the deadline. A timeout kills the whole
        session because a half-consumed response stream cannot be resynchronised.
        """
        timed_out = threading.Event()

        def _kill_on_timeout():
            timed_out.set()
            if self._proc:
                self._proc.kill()

        watchdog = threading.Timer(self.timeout, _kill_on_timeout)
        watchdog.start()

        lines: list[str] = []
        try:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    break  # server exited or was killed
                lines.append(line)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # non-JSON chatter on stdout is not fatal
                if message.get("id") == expected_id:
                    return "".join(lines)
        finally:
            watchdog.cancel()

        # Falling out of the loop means the stream ended before the reply did.
        if timed_out.is_set():
            self._dead_reason = f"timed out after {self.timeout}s"
            self.logger.error("MCP tool call timed out after %ss", self.timeout)
        else:
            returncode = self._proc.poll() if self._proc else None
            self._dead_reason = f"server exited with code {returncode}"
            self.logger.error("MCP server exited (code %s) before replying to id %s", returncode, expected_id)
        stderr_tail = "".join(self._stderr_tail).strip()
        if stderr_tail:
            self.logger.error("Stderr: %s", stderr_tail[-2000:])
        return None

    def _drain_stderr(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line)
        except (OSError, ValueError):
            pass  # stream closed during shutdown


class BaseSimulatorTest:
    """Base class for all communication simulator tests"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_files = {}
        self.test_dir = None
        # Set by server_session(); when present, call_mcp_tool reuses that
        # process instead of spawning a fresh one per call.
        self._server_session: MCPServerSession | None = None

        # Configure logging first
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.python_path = self._get_python_path()

    def _get_python_path(self) -> str:
        """Get the Python path for the virtual environment"""
        current_dir = os.getcwd()

        # Try .venv first (modern convention)
        venv_python = os.path.join(current_dir, ".venv", "bin", "python")
        if os.path.exists(venv_python):
            return venv_python

        # Try venv as fallback
        venv_python = os.path.join(current_dir, "venv", "bin", "python")
        if os.path.exists(venv_python):
            return venv_python

        # Try .pal_venv as fallback
        pal_venv_python = os.path.join(current_dir, ".pal_venv", "bin", "python")
        if os.path.exists(pal_venv_python):
            return pal_venv_python

        # Fallback to system python if venv doesn't exist
        self.logger.warning("Virtual environment not found, using system python")
        return "python"

    def setup_test_files(self):
        """Create test files for the simulation"""
        # Test Python file
        python_content = '''"""
Sample Python module for testing MCP conversation continuity
"""

def fibonacci(n):
    """Calculate fibonacci number recursively"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    """Calculate factorial iteratively"""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

class Calculator:
    """Simple calculator class"""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
'''

        # Test configuration file
        config_content = """{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "testdb",
    "ssl": true
  },
  "cache": {
    "redis_url": "redis://localhost:6379",
    "ttl": 3600
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}"""

        # Create files in the current project directory
        current_dir = os.getcwd()
        self.test_dir = os.path.join(current_dir, "test_simulation_files")
        os.makedirs(self.test_dir, exist_ok=True)

        test_py = os.path.join(self.test_dir, "test_module.py")
        test_config = os.path.join(self.test_dir, "config.json")

        with open(test_py, "w") as f:
            f.write(python_content)
        with open(test_config, "w") as f:
            f.write(config_content)

        # Ensure absolute paths for MCP server compatibility
        self.test_files = {"python": os.path.abspath(test_py), "config": os.path.abspath(test_config)}
        self.logger.debug(f"Created test files with absolute paths: {list(self.test_files.values())}")

    def apply_client_defaults(self, tool_name: str, params: dict) -> dict:
        """Fill in arguments a real MCP client always supplies.

        ``chat`` requires ``working_directory_absolute_path`` (tools/chat.py),
        which every client has on hand and no simulator scenario was passing --
        so every ``chat`` call in the suite came back as a pydantic validation
        error rather than reaching a provider (issue #132).

        Defaulting it here rather than at ~70 call sites keeps the scenarios
        about what they are testing. It is deliberately narrow: only arguments a
        client is expected to supply on the caller's behalf belong here, so a
        genuinely new required field still fails loudly instead of being
        papered over.
        """
        if tool_name != "chat" or "working_directory_absolute_path" in params:
            return params
        return {**params, "working_directory_absolute_path": self.test_dir or os.getcwd()}

    @contextmanager
    def server_session(self, timeout: int = 3600):
        """Hold one server process open across every ``call_mcp_tool`` in the block.

        Required by any scenario that passes a ``continuation_id`` from one tool
        call to the next: threads live in the server's process memory, so the
        second call has to reach the same process as the first (issue #132).
        Calls made outside a session still get a fresh one-shot server each,
        which is what tests with no continuation want.
        """
        session = MCPServerSession(self.python_path, self.logger, timeout=timeout)
        previous, self._server_session = self._server_session, session
        try:
            yield session
        finally:
            self._server_session = previous
            session.close()

    def call_mcp_tool(self, tool_name: str, params: dict) -> tuple[str | None, str | None]:
        """Call an MCP tool over stdio, reusing the active session if there is one."""
        try:
            params = self.apply_client_defaults(tool_name, params)
            session = self._server_session
            if session is not None:
                session.ensure_started()
                expected_id = session.next_response_id()
                stdout = session.call_tool(tool_name, params)
            else:
                # No active session: one server for this one call, as before.
                with MCPServerSession(self.python_path, self.logger) as one_shot:
                    expected_id = one_shot.next_response_id()
                    stdout = one_shot.call_tool(tool_name, params)

            if stdout is None:
                return None, None

            response_data = self._parse_mcp_response(stdout, expected_id=expected_id)
            if not response_data:
                return None, None

            continuation_id = self._extract_continuation_id(response_data)
            return response_data, continuation_id

        except Exception as e:
            self.logger.error(f"MCP tool call failed: {e}")
            return None, None

    def _parse_mcp_response(self, stdout: str, expected_id: int = 2) -> str | None:
        """Parse MCP JSON-RPC response from stdout"""
        try:
            lines = stdout.strip().split("\n")
            for line in lines:
                if line.strip() and line.startswith("{"):
                    response = json.loads(line)
                    # Look for the tool call response with the expected ID
                    if response.get("id") == expected_id and "result" in response:
                        # Extract the actual content from the response
                        result = response["result"]
                        # Handle new response format with 'content' array
                        if isinstance(result, dict) and "content" in result:
                            content_array = result["content"]
                            if isinstance(content_array, list) and len(content_array) > 0:
                                return content_array[0].get("text", "")
                        # Handle legacy format
                        elif isinstance(result, list) and len(result) > 0:
                            return result[0].get("text", "")
                    elif response.get("id") == expected_id and "error" in response:
                        self.logger.error(f"MCP error: {response['error']}")
                        return None

            # If we get here, log all responses for debugging
            self.logger.warning(f"No valid tool call response found for ID {expected_id}")
            self.logger.warning(f"Full stdout: {stdout}")
            self.logger.warning(f"Total stdout lines: {len(lines)}")
            for i, line in enumerate(lines[:10]):  # Log first 10 lines
                self.logger.warning(f"Line {i}: {line[:100]}...")
            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse MCP response: {e}")
            self.logger.debug(f"Stdout that failed to parse: {stdout}")
            return None

    def _extract_continuation_id(self, response_text: str) -> str | None:
        """Extract continuation_id from response metadata"""
        try:
            # Parse the response text as JSON to look for continuation metadata
            response_data = json.loads(response_text)

            # Look for continuation_id in various places
            if isinstance(response_data, dict):
                # Check for direct continuation_id field (new workflow tools)
                if "continuation_id" in response_data:
                    return response_data["continuation_id"]

                # Check metadata
                metadata = response_data.get("metadata", {})
                if "thread_id" in metadata:
                    return metadata["thread_id"]

                # Check follow_up_request
                follow_up = response_data.get("follow_up_request", {})
                if follow_up and "continuation_id" in follow_up:
                    return follow_up["continuation_id"]

                # Check continuation_offer
                continuation_offer = response_data.get("continuation_offer", {})
                if continuation_offer and "continuation_id" in continuation_offer:
                    return continuation_offer["continuation_id"]

            self.logger.debug(f"No continuation_id found in response: {response_data}")
            return None

        except json.JSONDecodeError as e:
            self.logger.debug(f"Failed to parse response for continuation_id: {e}")
            return None

    def run_command(self, cmd: list[str], check: bool = True, capture_output: bool = False, **kwargs):
        """Run a shell command with logging"""
        if self.verbose:
            self.logger.debug(f"Running: {' '.join(cmd)}")

        return subprocess.run(cmd, check=check, capture_output=capture_output, **kwargs)

    def create_additional_test_file(self, filename: str, content: str) -> str:
        """Create an additional test file for mixed scenario testing"""
        if not hasattr(self, "test_dir") or not self.test_dir:
            raise RuntimeError("Test directory not initialized. Call setup_test_files() first.")

        file_path = os.path.join(self.test_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        # Return absolute path for MCP server compatibility
        return os.path.abspath(file_path)

    def cleanup_test_files(self):
        """Clean up test files"""
        if hasattr(self, "test_dir") and self.test_dir and os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)
            self.logger.debug(f"Removed test files directory: {self.test_dir}")

    # ============================================================================
    # Log Utility Methods (delegate to LogUtils)
    # ============================================================================

    def get_server_logs_since(self, since_time: str | None = None) -> str:
        """Get server logs from both main and activity log files."""
        return LogUtils.get_server_logs_since(since_time)

    def get_recent_server_logs(self, lines: int = 500) -> str:
        """Get recent server logs from the main log file."""
        return LogUtils.get_recent_server_logs(lines)

    def get_server_logs_subprocess(self, lines: int = 500) -> str:
        """Get server logs using subprocess (alternative method)."""
        return LogUtils.get_server_logs_subprocess(lines)

    def check_server_logs_for_errors(self, lines: int = 500) -> list[str]:
        """Check server logs for error messages."""
        return LogUtils.check_server_logs_for_errors(lines)

    def extract_conversation_usage_logs(self, logs: str) -> list[dict[str, int]]:
        """Extract token budget calculation information from logs."""
        return LogUtils.extract_conversation_usage_logs(logs)

    def extract_conversation_token_usage(self, logs: str) -> list[int]:
        """Extract conversation token usage values from logs."""
        return LogUtils.extract_conversation_token_usage(logs)

    def extract_thread_creation_logs(self, logs: str) -> list[dict[str, str]]:
        """Extract thread creation logs with parent relationships."""
        return LogUtils.extract_thread_creation_logs(logs)

    def extract_history_traversal_logs(self, logs: str) -> list[dict[str, any]]:
        """Extract conversation history traversal logs."""
        return LogUtils.extract_history_traversal_logs(logs)

    def validate_file_deduplication_in_logs(self, logs: str, tool_name: str, test_file: str) -> bool:
        """Validate that logs show file deduplication behavior."""
        return LogUtils.validate_file_deduplication_in_logs(logs, tool_name, test_file)

    def search_logs_for_pattern(self, pattern: str, logs: str | None = None, case_sensitive: bool = False) -> list[str]:
        """Search logs for a specific pattern."""
        return LogUtils.search_logs_for_pattern(pattern, logs, case_sensitive)

    def get_log_file_info(self) -> dict[str, dict[str, any]]:
        """Get information about log files."""
        return LogUtils.get_log_file_info()

    def run_test(self) -> bool:
        """Run the test - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run_test()")

    @property
    def test_name(self) -> str:
        """Get the test name - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement test_name property")

    @property
    def test_description(self) -> str:
        """Get the test description - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement test_description property")
