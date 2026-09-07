"""Execute configured CLI agents for the clink tool and parse output."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from clink.constants import (
    BASE_ENV_ALLOWLIST,
    BASE_ENV_PREFIX_ALLOWLIST,
    DEFAULT_STREAM_LIMIT,
)
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.parsers import BaseParser, ParsedCLIResponse, ParserError, get_parser
from utils.progress import format_duration, get_progress_reporter

logger = logging.getLogger("clink.agent")


@dataclass
class AgentOutput:
    """Container returned by CLI agents after successful execution."""

    parsed: ParsedCLIResponse
    sanitized_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    parser_name: str
    output_file_content: str | None = None


class CLIAgentError(RuntimeError):
    """Raised when a CLI agent fails (non-zero exit, timeout, parse errors)."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BaseCLIAgent:
    """Execute a configured CLI command and parse its output."""

    def __init__(self, client: ResolvedCLIClient):
        self.client = client
        self._parser: BaseParser = get_parser(client.parser)
        self._logger = logging.getLogger(f"clink.runner.{client.name}")

    async def run(
        self,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None = None,
        files: Sequence[str],
        images: Sequence[str],
    ) -> AgentOutput:
        """Run the configured CLI in an explicit working directory.

        With no `working_dir` configured the subprocess used to inherit the
        server's own cwd, so a repo-aware CLI silently saw whatever tree the MCP
        server happened to be launched in (issue #119). The default is now an
        empty per-run scratch directory.

        That is a real default rather than a placeholder: the `clink` tool passes
        files as *absolute* paths (`absolute_file_paths`), which resolve from any
        cwd, so nothing the caller asked for depends on the inherited tree. A
        relay target that genuinely needs to sit inside a repository gets one by
        setting `working_dir` in `conf/cli_clients/*.json`.
        """
        if self.client.working_dir:
            return await self._run_in_directory(
                str(self.client.working_dir),
                role=role,
                prompt=prompt,
                system_prompt=system_prompt,
                files=files,
                images=images,
            )

        with tempfile.TemporaryDirectory(prefix="clink-workspace-") as scratch_dir:
            return await self._run_in_directory(
                scratch_dir,
                role=role,
                prompt=prompt,
                system_prompt=system_prompt,
                files=files,
                images=images,
            )

    async def _run_in_directory(
        self,
        cwd: str,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None,
        files: Sequence[str],
        images: Sequence[str],
    ) -> AgentOutput:
        # Files and images are already embedded into the prompt by the tool; they are
        # accepted here only to keep parity with SimpleTool callers.
        _ = (files, images)
        # The runner simply executes the configured CLI command for the selected role.
        command = self._build_command(role=role, system_prompt=system_prompt)
        env = self._build_environment()

        # Resolve executable path for cross-platform compatibility (especially Windows)
        executable_name = command[0]
        resolved_executable = shutil.which(executable_name)
        if resolved_executable is None:
            raise CLIAgentError(
                f"Executable '{executable_name}' not found in PATH for CLI '{self.client.name}'. "
                f"Ensure the command is installed and accessible."
            )
        command[0] = resolved_executable

        sanitized_command = list(command)

        limit = DEFAULT_STREAM_LIMIT

        stdout_text = ""
        stderr_text = ""
        output_file_content: str | None = None
        start_time = time.monotonic()

        output_file_path: Path | None = None
        command_with_output_flag = list(command)

        if self.client.output_to_file:
            fd, tmp_path = tempfile.mkstemp(prefix="clink-", suffix=".json")
            os.close(fd)
            output_file_path = Path(tmp_path)
            flag_template = self.client.output_to_file.flag_template
            try:
                rendered_flag = flag_template.format(path=str(output_file_path))
            except KeyError as exc:  # pragma: no cover - defensive
                raise CLIAgentError(f"Invalid output flag template '{flag_template}': missing placeholder {exc}")
            command_with_output_flag.extend(shlex.split(rendered_flag))
            sanitized_command = list(command_with_output_flag)

        self._logger.debug("Executing CLI command: %s", " ".join(sanitized_command))
        self._logger.debug("Working directory: %s", cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command_with_output_flag,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=limit,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CLIAgentError(f"Executable not found for CLI '{self.client.name}': {exc}") from exc

        # Report against the timeout, not just elapsed: an external CLI agent is
        # fully opaque while it runs, and "3m10s / 10m" tells the caller both that
        # it is alive and how much rope is left before it is killed.
        progress = get_progress_reporter()
        budget = format_duration(self.client.timeout_seconds)
        role_label = f" · {role.name}" if role.name else ""

        try:
            async with progress.heartbeat(f"clink · {self.client.name} CLI{role_label} · running · budget {budget}"):
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(prompt.encode("utf-8")),
                    timeout=self.client.timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise CLIAgentError(
                f"CLI '{self.client.name}' timed out after {self.client.timeout_seconds} seconds",
                returncode=None,
            ) from exc

        duration = time.monotonic() - start_time
        return_code = process.returncode
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        await progress.update(
            f"clink · {self.client.name} CLI{role_label} · exited {return_code} in {format_duration(duration)}"
        )

        if output_file_path and output_file_path.exists():
            output_file_content = output_file_path.read_text(encoding="utf-8", errors="replace")
            if self.client.output_to_file and self.client.output_to_file.cleanup:
                try:
                    output_file_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

            if output_file_content and not stdout_text.strip():
                stdout_text = output_file_content

        if return_code != 0:
            recovered = self._recover_from_error(
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                sanitized_command=sanitized_command,
                duration_seconds=duration,
                output_file_content=output_file_content,
            )
            if recovered is not None:
                return recovered

        if return_code != 0:
            raise CLIAgentError(
                f"CLI '{self.client.name}' exited with status {return_code}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        try:
            parsed = self._parser.parse(stdout_text, stderr_text)
        except ParserError as exc:
            raise CLIAgentError(
                f"Failed to parse output from CLI '{self.client.name}': {exc}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            ) from exc

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )

    def _build_command(self, *, role: ResolvedCLIRole, system_prompt: str | None) -> list[str]:
        base = list(self.client.executable)
        base.extend(self.client.internal_args)
        base.extend(self.client.config_args)
        base.extend(role.role_args)

        return base

    def _build_environment(self) -> dict[str, str]:
        """Build the subprocess environment from an allowlist rather than a copy.

        `os.environ.copy()` handed a relayed CLI every provider credential the
        server holds — relaying to `codex` passed it GEMINI_API_KEY, XAI_API_KEY,
        DIAL_API_KEY and the rest (issue #119). Only three groups get through now:

        1. `BASE_ENV_ALLOWLIST` / `BASE_ENV_PREFIX_ALLOWLIST` — what any CLI needs
           to run and reach the network. `HOME` is in there deliberately: it is
           where `claude`/`codex`/`gemini` keep their own subscription credentials,
           so a synthetic HOME would break the auth these relays rely on.
        2. `client.env_passthrough` — the prefixes of the target vendor's own
           variables, so each CLI still sees its own API key and nothing else.
        3. `client.env` — the explicit per-client block from
           `conf/cli_clients/*.json`, unchanged, and applied last so an operator
           can still override or inject anything.
        """
        allowed_prefixes = BASE_ENV_PREFIX_ALLOWLIST + tuple(self.client.env_passthrough)
        env = {
            key: value
            for key, value in os.environ.items()
            if key in BASE_ENV_ALLOWLIST or key.startswith(allowed_prefixes)
        }
        env.update(self.client.env)
        return env

    # ------------------------------------------------------------------
    # Error recovery hooks
    # ------------------------------------------------------------------

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        """Hook for subclasses to convert CLI errors into successful outputs.

        Return an AgentOutput to treat the failure as success, or None to signal
        that normal error handling should proceed.
        """

        return None
