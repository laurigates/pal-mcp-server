"""The pal-mcp-server tombstone package and its one-time publish job (issue #95).

The project moved to palaver-mcp-server at 13.0.0. packaging/pal-mcp-server-tombstone/
builds a final pal-mcp-server 13.0.0 that depends on palaver and keeps the old
command, and the publish-tombstone job in release-please.yml uploads it for v13.0.0
only. Both are meant to be deleted once that release has shipped.
"""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; pytest depends on tomli there
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
TOMBSTONE_DIR = REPO_ROOT / "packaging" / "pal-mcp-server-tombstone"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-please.yml"
TOMBSTONE_RELEASE = (13, 0, 0)


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _version_tuple(version: str) -> tuple[int, ...]:
    """'13.0.1' -> (13, 0, 1); a pre-release suffix on a part is ignored."""
    return tuple(int(re.match(r"\d+", part).group()) for part in version.split(".")[:3])


def test_tombstone_removed_after_13_0_0_ships():
    """Tripwire (#95): once the version passes 13.0.0, the tombstone has shipped.

    The tombstone is published exactly once, by the publish-tombstone job for
    v13.0.0. After release-please bumps past 13.0.0 the job can never run again
    and the packaging directory is dead weight. The only resolution is to
    delete both; this test then passes and can be deleted with them.
    """
    version = _load_toml(REPO_ROOT / "pyproject.toml")["project"]["version"]
    if _version_tuple(version) <= TOMBSTONE_RELEASE:
        return

    leftovers = []
    if TOMBSTONE_DIR.exists():
        leftovers.append(str(TOMBSTONE_DIR.relative_to(REPO_ROOT)))
    if re.search(r"^  publish-tombstone:", RELEASE_WORKFLOW.read_text(), re.MULTILINE):
        leftovers.append(f"publish-tombstone job in {RELEASE_WORKFLOW.relative_to(REPO_ROOT)}")
    assert not leftovers, (
        f"pyproject.toml is at {version}, past the 13.0.0 tombstone release, but "
        f"{' and '.join(leftovers)} still exist. pal-mcp-server 13.0.0 has shipped; "
        "delete packaging/pal-mcp-server-tombstone/, the publish-tombstone job, and this test (#95)."
    )


def test_tombstone_depends_on_palaver_with_a_floor():
    """A floor, not a pin, so the single tombstone release keeps resolving the newest palaver."""
    project = _load_toml(TOMBSTONE_DIR / "pyproject.toml")["project"]
    assert project["name"] == "pal-mcp-server"
    assert project["version"] == "13.0.0"
    assert project["dependencies"] == ["palaver-mcp-server>=13.0.0"]


def test_tombstone_keeps_the_old_command():
    project = _load_toml(TOMBSTONE_DIR / "pyproject.toml")["project"]
    assert project["scripts"] == {"pal-mcp-server": "pal_mcp_server_tombstone:main"}


def test_tombstone_requires_python_matches_main_package():
    tombstone = _load_toml(TOMBSTONE_DIR / "pyproject.toml")["project"]
    main = _load_toml(REPO_ROOT / "pyproject.toml")["project"]
    assert tombstone["requires-python"] == main["requires-python"]


def test_main_package_does_not_ship_the_old_command():
    """Two installed distributions providing one console script overwrite each other's launcher."""
    project = _load_toml(REPO_ROOT / "pyproject.toml")["project"]
    assert project["name"] == "palaver-mcp-server"
    assert "pal-mcp-server" not in project["scripts"]


def test_main_package_build_excludes_packaging_dir():
    """setuptools' find include list must not pick up the tombstone's directory."""
    include = _load_toml(REPO_ROOT / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]["include"]
    assert not any(pattern.startswith("packaging") for pattern in include)


def test_tombstone_notice_goes_to_stderr_only(monkeypatch, capsys):
    """stdout is the MCP stdio channel; a byte there breaks the protocol handshake."""
    import importlib.util
    import sys
    import types

    spec = importlib.util.spec_from_file_location(
        "pal_mcp_server_tombstone", TOMBSTONE_DIR / "pal_mcp_server_tombstone.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(run=lambda: calls.append("run")))
    module.main()

    out, err = capsys.readouterr()
    assert out == ""
    assert "palaver-mcp-server" in err
    assert calls == ["run"]
