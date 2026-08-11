"""
Regression tests for issue #77 — pseudo-filesystem reads bypassing path policy.

Two independent holes let a caller-supplied path read the server's own
process environment (every provider API key) on Linux:

1. ``DANGEROUS_SYSTEM_PATHS`` was a denylist that omitted every
   pseudo-filesystem, so ``/proc/self/environ`` passed validation.
2. Both size guards trusted ``stat().st_size``. procfs reports 0 for
   files with content, so such a file was free to the MCP-boundary gate
   *and* under the ``max_size`` cap, while ``read()`` returned everything.

(2) is the one that generalises: it holds for any pseudo-file nobody
thought to add to (1).
"""

from pathlib import Path

import pytest

from utils.file_utils import estimate_file_tokens, read_file_content
from utils.security_config import is_dangerous_path


@pytest.mark.parametrize(
    "path",
    [
        "/proc",
        "/proc/self/environ",
        "/proc/self/cmdline",
        "/proc/1/environ",
        "/sys",
        "/sys/class/net",
        "/dev",
        "/dev/null",
        "/boot",
        "/lib",
        "/sbin",
        "/run",
    ],
)
def test_pseudo_filesystems_are_dangerous(path):
    assert is_dangerous_path(Path(path)) is True


@pytest.mark.parametrize("path", ["/etc/passwd", "/usr/bin/env", "/var/log", "/root"])
def test_previously_blocked_paths_stay_blocked(path):
    """The added entries must not have displaced the original policy."""
    assert is_dangerous_path(Path(path)) is True


def test_project_paths_are_still_allowed(tmp_path):
    """The denylist must not swallow ordinary caller project files.

    PAL exists to read the caller's files, which live outside the server's
    own root -- a containment allowlist would break the tool entirely.
    """
    f = tmp_path / "app.py"
    f.write_text("print('hello')\n")
    assert is_dangerous_path(f) is False
    assert estimate_file_tokens(str(f)) > 0


def test_pseudo_path_estimates_as_zero_not_readable():
    """A blocked path contributes nothing to the size gate."""
    assert estimate_file_tokens("/proc/self/environ") == 0


def test_read_is_capped_even_when_stat_size_lies(tmp_path, monkeypatch):
    """The read cap must not trust st_size.

    Simulates a procfs-style file: stat reports 0 bytes, the content is
    large. Before the fix this sailed past `file_size > max_size` and was
    read in full.
    """
    victim = tmp_path / "fake_procfs_entry.txt"
    payload = "SECRET=" + ("A" * 50_000)
    victim.write_text(payload)

    real_stat = Path.stat

    def lying_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self.name == "fake_procfs_entry.txt":

            class _Lying:
                st_size = 0
                st_mtime = result.st_mtime
                st_mode = result.st_mode

            return _Lying()
        return result

    monkeypatch.setattr(Path, "stat", lying_stat)

    content, _tokens = read_file_content(str(victim), max_size=1_000)

    assert "FILE TOO LARGE" in content
    assert "A" * 1_000 not in content
    assert "SECRET=" not in content
