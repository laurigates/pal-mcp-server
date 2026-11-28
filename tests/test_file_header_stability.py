"""File headers carry no modification time (issue #195, Stage B).

A ``(Last modified: …)`` stamp changed whenever a file was touched or checked
out, even with identical content, and so invalidated every cached prompt byte
after that file. The headers now depend on the path and content only.
"""

from __future__ import annotations

import os

from utils.file_utils import read_file_content


def test_header_is_unchanged_by_touching_the_file(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("VALUE = 1\n", encoding="utf-8")
    before, _ = read_file_content(str(path))

    os.utime(path, (1_000_000_000, 1_000_000_000))
    after, _ = read_file_content(str(path))

    assert before == after
    assert "Last modified" not in before


def test_too_large_header_has_no_modification_time(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 200, encoding="utf-8")

    content, _ = read_file_content(str(path), max_size=100)

    assert "FILE TOO LARGE" in content
    assert "Last modified" not in content
