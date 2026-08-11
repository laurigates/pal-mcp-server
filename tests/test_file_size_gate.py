"""
Tests for the MCP-boundary file-size gate (``check_total_file_size``).

The gate rejects a tool call outright when the estimated size of the
attached ``absolute_file_paths`` exceeds the budget, rather than letting
``read_files`` embed a subset. Two properties matter and had no coverage:

1. The gate's threshold must match the budget the embedding layer
   actually uses (``TokenAllocation.file_tokens``). A gate that is
   stricter than the embedder rejects file sets that would have been
   embedded in full.
2. The rejection must tell the caller *which* files are expensive.
   "Too large, pick fewer" is not actionable without per-file sizes.
"""

import pytest

from utils.file_utils import check_total_file_size
from utils.model_context import TokenAllocation


class StubModelContext:
    """Minimal ModelContext stand-in with a caller-chosen context window."""

    def __init__(self, model_name: str, context_window: int):
        self.model_name = model_name
        self._context_window = context_window

    def calculate_token_allocation(self, reserved_for_response: int | None = None) -> TokenAllocation:
        total = self._context_window
        content_ratio, file_ratio, history_ratio = (0.6, 0.3, 0.5) if total < 300_000 else (0.8, 0.4, 0.4)
        content = int(total * content_ratio)
        return TokenAllocation(
            total_tokens=total,
            content_tokens=content,
            response_tokens=int(total * (1 - content_ratio)),
            file_tokens=int(content * file_ratio),
            history_tokens=int(content * history_ratio),
        )


@pytest.fixture
def stub_context(monkeypatch):
    """Patch ModelContext so the gate uses a context window we control."""

    def _install(context_window: int):
        import utils.model_context

        monkeypatch.setattr(
            utils.model_context,
            "ModelContext",
            lambda name, *a, **kw: StubModelContext(name, context_window),
        )

    return _install


def _make_file(tmp_path, name: str, estimated_tokens: int):
    """Create a .py file whose size-based estimate is ``estimated_tokens``."""
    path = tmp_path / name
    path.write_text("x" * int(estimated_tokens * 3.5))  # .py ratio is 3.5
    return str(path)


# Kimi (and every other 262K model): 262144 * 0.6 content * 0.3 files
KIMI_CONTEXT = 262_144
KIMI_FILE_TOKENS = int(int(KIMI_CONTEXT * 0.6) * 0.3)  # 47,185


def test_gate_threshold_equals_embedding_budget(tmp_path, stub_context):
    """The gate must not reject what the embedder would have accepted."""
    stub_context(KIMI_CONTEXT)
    just_under = _make_file(tmp_path, "a.py", KIMI_FILE_TOKENS - 500)

    assert check_total_file_size([just_under], "kimi") is None


def test_gate_rejects_above_embedding_budget(tmp_path, stub_context):
    stub_context(KIMI_CONTEXT)
    too_big = _make_file(tmp_path, "a.py", KIMI_FILE_TOKENS + 5_000)

    result = check_total_file_size([too_big], "kimi")

    assert result is not None
    assert result["status"] == "code_too_large"
    assert result["metadata"]["limit"] == KIMI_FILE_TOKENS


def test_gate_admits_the_band_the_old_haircut_rejected(tmp_path, stub_context):
    """Regression: a 0.6 haircut rejected 28,311-47,185 tokens needlessly."""
    stub_context(KIMI_CONTEXT)
    in_old_dead_band = _make_file(tmp_path, "a.py", 35_000)  # > 28,311, < 47,185

    assert check_total_file_size([in_old_dead_band], "kimi") is None


@pytest.mark.parametrize("context_window", [128_000, 262_144, 500_000, 1_048_576])
def test_gate_scales_without_cliffs(tmp_path, stub_context, context_window):
    """Gate is a fixed fraction of file_tokens at every context size."""
    stub_context(context_window)
    content_ratio, file_ratio = (0.6, 0.3) if context_window < 300_000 else (0.8, 0.4)
    file_tokens = int(int(context_window * content_ratio) * file_ratio)

    fits = _make_file(tmp_path, "fits.py", file_tokens - 200)
    assert check_total_file_size([fits], "m") is None

    over = _make_file(tmp_path, "over.py", file_tokens + 2_000)
    assert check_total_file_size([over], "m")["metadata"]["limit"] == file_tokens


def test_rejection_names_the_largest_files(tmp_path, stub_context):
    """The caller must learn which files to drop, not just that it failed."""
    stub_context(KIMI_CONTEXT)
    small = _make_file(tmp_path, "small.py", 1_000)
    huge = _make_file(tmp_path, "huge.py", KIMI_FILE_TOKENS)
    medium = _make_file(tmp_path, "medium.py", 8_000)

    result = check_total_file_size([small, huge, medium], "kimi")

    assert result is not None
    breakdown = result["metadata"]["largest_files"]
    # Sorted biggest-first so the caller can drop from the top.
    assert [entry["path"] for entry in breakdown] == [huge, medium, small]
    assert breakdown[0]["estimated_tokens"] > breakdown[1]["estimated_tokens"]
    # The human-readable message must surface the worst offender too.
    assert "huge.py" in result["content"]


def test_no_files_passes(stub_context):
    stub_context(KIMI_CONTEXT)
    assert check_total_file_size([], "kimi") is None


def test_unresolved_model_is_rejected(tmp_path, stub_context):
    stub_context(KIMI_CONTEXT)
    f = _make_file(tmp_path, "a.py", 10)
    with pytest.raises(ValueError, match="unresolved model"):
        check_total_file_size([f], "auto")
