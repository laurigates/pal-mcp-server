"""Pin the MCP tool surface.

Three properties, each a deterministic form of a constraint from issues #90/#91:

* the registered tool-name set is exactly the 19 names below (#91: one name per
  capability; a deferring client routes on names alone),
* every tool description fits the eager channel budget (#90: descriptions are
  what such a client pays every session), and
* ``scripts/tool_surface.py`` round-trips (``snapshot`` then ``check`` passes)
  and ``measure --json`` keeps the keys the README numbers are quoted from.

The script is exercised as a subprocess so the registry is built exactly the
way the script builds it (``DISABLED_TOOLS`` cleared before ``server`` is
imported), independent of whatever the test process imported earlier.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tool_surface.py"

EXPECTED_TOOL_NAMES = frozenset(
    {
        "analyze",
        "apilookup",
        "challenge",
        "chat",
        "clink",
        "codereview",
        "consensus",
        "debug",
        "docgen",
        "jules",
        "listmodels",
        "planner",
        "precommit",
        "refactor",
        "secaudit",
        "testgen",
        "thinkdeep",
        "tracer",
        "version",
    }
)
assert len(EXPECTED_TOOL_NAMES) == 19

# Longest description any tool may ship. tools/tracer.py sits at the cap because its
# tests pin four substrings; everything else has headroom.
MAX_DESCRIPTION_CHARS = 220


# Only what the interpreter needs to start. Provider credentials and model
# restrictions are deliberately absent: they append roster text to the model
# parameter's description, which would make the measured numbers depend on the
# developer's shell rather than on the source.
_PASSTHROUGH_ENV = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "PYTHONPATH", "SYSTEMROOT")


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
    env.update({"DISABLED_TOOLS": "", "LOG_LEVEL": "ERROR", "PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def measured_surface() -> dict:
    result = _run_script("measure", "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def registered_tools() -> dict:
    """server.TOOLS with DISABLED_TOOLS cleared, built the way scripts/tool_surface.py builds it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISABLED_TOOLS", "")
        mp.syspath_prepend(str(REPO_ROOT))
        import importlib

        import server

        importlib.reload(server)
        tools = dict(server.TOOLS)
    return tools


class TestToolNames:
    def test_registered_tool_names_are_exactly_the_nineteen(self, registered_tools):
        assert set(registered_tools) == EXPECTED_TOOL_NAMES

    def test_measured_tool_names_match_registry(self, measured_surface):
        assert {row["tool"] for row in measured_surface["per_tool"]} == EXPECTED_TOOL_NAMES
        assert measured_surface["all"]["tools"] == 19


class TestDescriptionBudget:
    def test_every_description_fits_the_eager_budget(self, registered_tools):
        too_long = {
            name: len(tool.get_description())
            for name, tool in registered_tools.items()
            if len(tool.get_description()) > MAX_DESCRIPTION_CHARS
        }
        assert not too_long, f"descriptions over {MAX_DESCRIPTION_CHARS} chars: {too_long}"

    def test_measured_description_chars_match_registry(self, registered_tools, measured_surface):
        measured = {row["tool"]: row["description_chars"] for row in measured_surface["per_tool"]}
        live = {name: len(tool.get_description()) for name, tool in registered_tools.items()}
        assert measured == live


class TestToolSurfaceScript:
    def test_measure_json_has_expected_keys(self, measured_surface):
        assert set(measured_surface) == {"per_tool", "all", "enabled", "disabled"}
        for totals in (measured_surface["all"], measured_surface["enabled"]):
            assert set(totals) == {
                "tools",
                "name_chars",
                "description_chars",
                "schema_chars",
                "eager_chars",
                "full_chars",
            }
            assert totals["eager_chars"] == totals["name_chars"] + totals["description_chars"]
            assert totals["full_chars"] == totals["eager_chars"] + totals["schema_chars"]
        for row in measured_surface["per_tool"]:
            assert set(row) == {"tool", "name_chars", "description_chars", "schema_chars"}
        assert measured_surface["disabled"] == []

    def test_preset_disables_the_shipped_default(self, measured_surface):
        result = _run_script("measure", "--preset", "--json")
        assert result.returncode == 0, result.stderr
        preset = json.loads(result.stdout)
        assert preset["disabled"] == ["analyze", "docgen", "refactor", "secaudit", "testgen", "tracer"]
        assert preset["enabled"]["tools"] == 13
        assert preset["all"] == measured_surface["all"]
        assert preset["enabled"]["full_chars"] < preset["all"]["full_chars"]

    def test_snapshot_then_check_round_trips(self, tmp_path):
        snapshot = tmp_path / "surface.json"
        wrote = _run_script("snapshot", "--out", str(snapshot))
        assert wrote.returncode == 0, wrote.stderr
        assert set(json.loads(snapshot.read_text())) == EXPECTED_TOOL_NAMES

        checked = _run_script("check", "--against", str(snapshot))
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert "matches" in checked.stdout

    def test_check_fails_on_structural_drift(self, tmp_path):
        snapshot = tmp_path / "surface.json"
        assert _run_script("snapshot", "--out", str(snapshot)).returncode == 0
        data = json.loads(snapshot.read_text())
        data["chat"]["properties"].pop("prompt")
        snapshot.write_text(json.dumps(data))

        checked = _run_script("check", "--against", str(snapshot))
        assert checked.returncode == 1
        assert "chat.properties.prompt" in checked.stdout
