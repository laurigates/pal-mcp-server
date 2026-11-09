#!/usr/bin/env python3
"""Build the Claude Desktop extension bundle: dist/pal-mcp-server-<version>.mcpb.

The bundle uses the MCPB ``uv`` server type (manifest_version 0.4): it ships
the source, ``pyproject.toml`` and ``uv.lock``, and the host installs the
dependencies with uv on first launch, so one bundle serves every platform.

Steps:
  1. manifest.json version must equal pyproject.toml version (release-please
     bumps both; see extra-files in release-please-config.json).
  2. Stage the git-tracked files under BUNDLE_PATHS into a temp dir. An
     allowlist of tracked files, so neither a local .env nor a new top-level
     file can ride along unreviewed.
  3. ``mcpb validate`` + ``mcpb pack`` into dist/.

Run scripts/smoke_mcpb.py on the result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
MCPB_CLI = ["npx", "--yes", "@anthropic-ai/mcpb@2.1.2"]

# What the server needs at runtime, and what `uv run` needs to install the
# project (pyproject's license-files names LICENSE and NOTICE).
BUNDLE_PATHS = [
    "manifest.json",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "NOTICE",
    "README.md",
    "server.py",
    "config.py",
    "clink",
    "conf",
    "providers",
    "systemprompts",
    "tools",
    "utils",
]


def fail(message: str, extra: str = "") -> None:
    print(f"❌ build_mcpb failed: {message}", file=sys.stderr)
    if extra:
        print(extra, file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"{' '.join(cmd)} exited {result.returncode}", f"{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> None:
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text())
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    if manifest["version"] != version:
        fail(
            f"manifest.json version {manifest['version']} != pyproject.toml version {version}",
            "release-please bumps both via extra-files in release-please-config.json.",
        )

    tracked = run(["git", "ls-files", "-z", "--", *BUNDLE_PATHS]).split("\0")
    tracked = [path for path in tracked if path]
    # manifest.json is new on a branch until committed; stage it regardless.
    if "manifest.json" not in tracked:
        tracked.append("manifest.json")

    with tempfile.TemporaryDirectory(prefix="pal-mcpb-") as tmp:
        staging = Path(tmp) / "bundle"
        for rel in tracked:
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel, dest)

        entry = staging / manifest["server"]["entry_point"]
        if not entry.is_file():
            fail(f"entry_point {manifest['server']['entry_point']} not staged")

        dist = REPO_ROOT / "dist"
        dist.mkdir(exist_ok=True)
        output = dist / f"pal-mcp-server-{version}.mcpb"
        run([*MCPB_CLI, "validate", str(staging / "manifest.json")])
        run([*MCPB_CLI, "pack", str(staging), str(output)])

    print(f"✅ Built {output} ({len(tracked)} files)")


if __name__ == "__main__":
    main()
