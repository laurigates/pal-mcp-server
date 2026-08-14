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

from .log_utils import LogUtils


class BaseSimulatorTest:
    """Base class for all communication simulator tests"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_files = {}
        self.test_dir = None

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

    def call_mcp_tool(self, tool_name: str, params: dict) -> tuple[str | None, str | None]:
        """Call an MCP tool via standalone server"""
        try:
            # Prepare the MCP initialization and tool call sequence
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "communication-simulator", "version": "1.0.0"},
                },
            }

            # Send initialized notification
            initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}

            # Prepare the tool call request
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
            }  # Combine all messages
            messages = [
                json.dumps(init_request, ensure_ascii=False),
                json.dumps(initialized_notification, ensure_ascii=False),
                json.dumps(tool_request, ensure_ascii=False),
            ]

            # Join with newlines as MCP expects
            input_data = "\n".join(messages) + "\n"

            # Call the standalone MCP server directly
            server_cmd = [self.python_path, "server.py"]

            self.logger.debug(f"Calling MCP tool {tool_name} with proper initialization")

            stdout = self._exchange_over_stdio(server_cmd, input_data, expected_id=2)
            if stdout is None:
                return None, None

            # Parse the response - look for the tool call response
            response_data = self._parse_mcp_response(stdout, expected_id=2)
            if not response_data:
                return None, None

            # Extract continuation_id if present
            continuation_id = self._extract_continuation_id(response_data)

            return response_data, continuation_id

        except Exception as e:
            self.logger.error(f"MCP tool call failed: {e}")
            return None, None

    def _exchange_over_stdio(self, server_cmd: list[str], input_data: str, expected_id: int, timeout: int = 3600):
        """Send the JSON-RPC batch and read until the reply to ``expected_id`` arrives.

        stdin must stay open until then. The stdio server treats EOF as shutdown, so
        writing the batch and closing (what subprocess.run does) cancels any handler
        that is still awaiting — the server exits 0 having answered only `initialize`.
        Tools that return without I/O beat the race; every tool that calls a provider
        loses it, which is why the whole suite failed a few hundred ms in.
        """
        proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # readline() blocks with no timeout of its own; killing the process is what
        # unblocks it, so the watchdog owns the deadline.
        timed_out = threading.Event()

        def _kill_on_timeout():
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(timeout, _kill_on_timeout)
        watchdog.start()

        lines: list[str] = []
        try:
            proc.stdin.write(input_data)
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line:
                    break  # server exited or was killed
                lines.append(line)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # non-JSON chatter on stdout is not fatal
                if message.get("id") == expected_id:
                    break
        finally:
            watchdog.cancel()
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
            stderr = proc.stderr.read() if proc.stderr else ""
            proc.stderr.close() if proc.stderr else None
            proc.stdout.close() if proc.stdout else None

        if timed_out.is_set():
            self.logger.error(f"MCP tool call timed out after {timeout}s")
            return None

        if proc.returncode not in (0, None):
            self.logger.error(f"Standalone server failed with return code {proc.returncode}")
            if stderr:
                self.logger.error(f"Stderr: {stderr.strip()[-2000:]}")

        return "".join(lines)

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
