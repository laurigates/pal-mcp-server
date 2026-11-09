#!/usr/bin/env python3
"""Smoke test for the Claude Desktop extension bundle.

Unzips dist/pal-mcp-server-<version>.mcpb into an empty directory, resolves the
manifest's mcp_config the way a host does (``${__dirname}``, ``${user_config.*}``
with every key left at its default, so no provider is configured) and fails on
any unresolved ``${...}`` placeholder. It then launches the server from an
unrelated cwd with an empty HOME, so nothing outside the bundle (the repo's
.venv, a .env, a user config) can satisfy an import or a setting, completes the
MCP ``initialize`` handshake and requires a non-empty ``tools/list``.

uv's cache and managed-Python dirs are passed through so the first-launch
install reuses already-downloaded wheels instead of refetching them.

Usage: uv run python scripts/smoke_mcpb.py [path/to/bundle.mcpb]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 300  # first launch resolves and installs every dependency
PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")


def fail(message: str, extra: str = "") -> None:
    print(f"❌ smoke_mcpb: {message}", file=sys.stderr)
    if extra:
        print(extra, file=sys.stderr)
    sys.exit(1)


def resolve(value: str, extension_dir: Path, user_config: dict[str, str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "__dirname":
            return str(extension_dir)
        if key.startswith("user_config.") and key.removeprefix("user_config.") in user_config:
            return user_config[key.removeprefix("user_config.")]
        return match.group(0)  # left in place; reported below

    return PLACEHOLDER.sub(substitute, value)


def uv_dir(*args: str) -> str:
    return subprocess.run(["uv", *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "dist" / f"pal-mcp-server-{version}.mcpb"
    if not bundle.is_file():
        fail(f"bundle not found: {bundle} — run `just mcpb-build` first")

    with tempfile.TemporaryDirectory(prefix="pal-mcpb-smoke-") as tmp:
        tmp_path = Path(tmp)
        extension_dir = tmp_path / "extension"
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(extension_dir)

        manifest = json.loads((extension_dir / "manifest.json").read_text())
        user_config = {key: str(spec.get("default", "")) for key, spec in manifest.get("user_config", {}).items()}
        mcp_config = manifest["server"]["mcp_config"]
        command = resolve(mcp_config["command"], extension_dir, user_config)
        args = [resolve(arg, extension_dir, user_config) for arg in mcp_config.get("args", [])]
        config_env = {k: resolve(v, extension_dir, user_config) for k, v in mcp_config.get("env", {}).items()}

        unresolved = PLACEHOLDER.findall(json.dumps([command, args, config_env]))
        if unresolved:
            fail(f"unresolved placeholders: {', '.join(sorted(set(unresolved)))}")

        env = {
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or uv_dir("cache", "dir"),
            "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR") or uv_dir("python", "dir"),
            **config_env,
        }
        child = subprocess.Popen(
            [command, *args],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdin and child.stdout and child.stderr  # all three are PIPE
        stdin, stdout, stderr = child.stdin, child.stdout, child.stderr
        stderr_chunks: list[str] = []
        threading.Thread(target=lambda: stderr_chunks.extend(stderr), daemon=True).start()
        timer = threading.Timer(TIMEOUT_S, child.kill)
        timer.start()

        def send(message: dict) -> None:
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()

        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke_mcpb", "version": "0"},
                },
            }
        )
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

        try:
            for line in stdout:
                message = json.loads(line)
                if message.get("id") != 2:
                    continue
                tools = message.get("result", {}).get("tools", [])
                if not tools:
                    fail("tools/list returned no tools", line)
                print(f"✅ MCPB smoke test passed: {len(tools)} tools from {bundle}")
                return
        finally:
            timer.cancel()
            child.kill()
            child.wait()

        fail(
            f"server exited (code {child.returncode}) without answering tools/list within {TIMEOUT_S}s",
            "".join(stderr_chunks)[-4000:],
        )


if __name__ == "__main__":
    main()
