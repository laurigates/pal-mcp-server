"""Pin manifest.json (the Claude Desktop extension) to the rest of the repo.

The bundle itself is built and launched by the `mcpb` job in test.yml; these
checks catch the cheap drifts in the unit tier, on every Python version.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((REPO_ROOT / "manifest.json").read_text())


def _pyproject_version() -> str:
    # tomllib is 3.11+; the unit tier also runs on 3.10.
    match = re.search(r'^version = "([^"]+)"', (REPO_ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    assert match
    return match.group(1)


def test_manifest_version_matches_pyproject():
    # release-please bumps both through extra-files in release-please-config.json.
    assert MANIFEST["version"] == _pyproject_version()


def test_release_please_bumps_manifest_version():
    config = json.loads((REPO_ROOT / "release-please-config.json").read_text())
    extra_files = config["packages"]["."].get("extra-files", [])
    assert {"type": "json", "path": "manifest.json", "jsonpath": "$.version"} in extra_files


def test_every_user_config_key_reaches_the_server_env():
    env = MANIFEST["server"]["mcp_config"]["env"]
    referenced = {m for value in env.values() for m in re.findall(r"\$\{user_config\.([^}]+)\}", value)}
    assert referenced == set(MANIFEST["user_config"])


def test_entry_point_exists():
    assert (REPO_ROOT / MANIFEST["server"]["entry_point"]).is_file()
