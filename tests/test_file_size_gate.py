"""
Tests for the MCP-boundary file-size gate (``check_total_file_size``).

The gate rejects a tool call outright when the estimated size of the
attached ``absolute_file_paths`` exceeds the budget, rather than letting
``read_files`` embed a subset. Three properties matter and had no coverage:

1. The gate's threshold must equal the budget the embedding layer actually
   uses -- both derive it from ``TokenAllocation.effective_file_budget``.
   A gate stricter than the embedder rejects file sets that would have
   been embedded in full; a laxer one breaks its own all-or-nothing
   promise.
2. On a continuation the budget is what conversation history left over,
   which may be larger *or* smaller than the static file allocation.
   Both directions are tested -- capping with ``min()`` would silently
   re-introduce (1).
3. The rejection must tell the caller *which* files are expensive and
   *why* the limit is what it is.
"""

import pytest

from utils.file_utils import check_total_file_size
from utils.model_context import DEFAULT_FILE_RESERVE_TOKENS, TokenAllocation


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
KIMI_BUDGET = KIMI_FILE_TOKENS - DEFAULT_FILE_RESERVE_TOKENS  # 46,185
KIMI_CONTENT_TOKENS = int(KIMI_CONTEXT * 0.6)  # 157,286


def test_gate_threshold_equals_embedding_budget(tmp_path, stub_context):
    """The gate must not reject what the embedder would have accepted."""
    stub_context(KIMI_CONTEXT)
    just_under = _make_file(tmp_path, "a.py", KIMI_BUDGET - 500)

    assert check_total_file_size([just_under], "kimi") is None


def test_gate_rejects_above_embedding_budget(tmp_path, stub_context):
    stub_context(KIMI_CONTEXT)
    too_big = _make_file(tmp_path, "a.py", KIMI_BUDGET + 5_000)

    result = check_total_file_size([too_big], "kimi")

    assert result is not None
    assert result["status"] == "code_too_large"
    assert result["metadata"]["limit"] == KIMI_BUDGET
    assert result["metadata"]["bound_by"] == "model_file_allocation"


def test_gate_reserve_matches_the_embedder(stub_context):
    """The gate's limit is file_tokens minus the same reserve the embedder holds back."""
    stub_context(KIMI_CONTEXT)
    allocation = TokenAllocation(
        total_tokens=KIMI_CONTEXT,
        content_tokens=KIMI_CONTENT_TOKENS,
        response_tokens=0,
        file_tokens=KIMI_FILE_TOKENS,
        history_tokens=0,
    )
    assert allocation.effective_file_budget() == KIMI_BUDGET


def test_long_history_shrinks_the_gate(tmp_path, stub_context):
    """A continuation whose history ate the budget must not be waved through."""
    stub_context(KIMI_CONTEXT)
    remaining = 20_000  # history consumed most of content_tokens
    # Comfortably under the static allocation, but over what actually remains.
    files = [_make_file(tmp_path, "a.py", 30_000)]

    assert check_total_file_size(files, "kimi") is None  # fresh call: fits
    result = check_total_file_size(files, "kimi", remaining_tokens=remaining)

    assert result is not None
    assert result["metadata"]["limit"] == remaining - DEFAULT_FILE_RESERVE_TOKENS
    assert result["metadata"]["bound_by"] == "conversation_history"


def test_short_history_does_not_shrink_the_gate(tmp_path, stub_context):
    """Regression: capping with min(file_tokens, remaining) would reject this.

    With a short history the leftover content budget exceeds the static file
    allocation, and the embedder deliberately uses the larger figure.
    """
    stub_context(KIMI_CONTEXT)
    remaining = KIMI_CONTENT_TOKENS - 10_000  # 147,286 -- well above file_tokens
    big = _make_file(tmp_path, "a.py", 100_000)  # > file_tokens, < remaining

    assert check_total_file_size([big], "kimi", remaining_tokens=remaining) is None


def test_rejection_explains_a_history_bound_limit(tmp_path, stub_context):
    """A small limit on a huge-context model must explain itself."""
    stub_context(KIMI_CONTEXT)
    files = [_make_file(tmp_path, "a.py", 30_000)]

    result = check_total_file_size(files, "kimi", remaining_tokens=5_000)

    assert "history" in result["content"].lower()
    assert "continuation_id" in result["content"]


def test_gate_admits_the_band_the_old_haircut_rejected(tmp_path, stub_context):
    """Regression: a 0.6 haircut rejected 28,311-47,185 tokens needlessly."""
    stub_context(KIMI_CONTEXT)
    in_old_dead_band = _make_file(tmp_path, "a.py", 35_000)  # > 28,311, < 46,185

    assert check_total_file_size([in_old_dead_band], "kimi") is None


@pytest.mark.parametrize("context_window", [128_000, 262_144, 500_000, 1_048_576])
def test_gate_scales_without_cliffs(tmp_path, stub_context, context_window):
    """Gate is a fixed fraction of file_tokens at every context size."""
    stub_context(context_window)
    content_ratio, file_ratio = (0.6, 0.3) if context_window < 300_000 else (0.8, 0.4)
    budget = int(int(context_window * content_ratio) * file_ratio) - DEFAULT_FILE_RESERVE_TOKENS

    fits = _make_file(tmp_path, "fits.py", budget - 200)
    assert check_total_file_size([fits], "m") is None

    over = _make_file(tmp_path, "over.py", budget + 2_000)
    assert check_total_file_size([over], "m")["metadata"]["limit"] == budget


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


def test_breakdown_does_not_leak_protected_paths(tmp_path, stub_context):
    """The rejection must not report sizes for paths policy would block.

    The gate runs at the MCP boundary, before any tool executes and so before
    read_file_content/expand_paths would have validated anything. Reporting a
    per-file size for /etc/passwd would turn the rejection into an existence
    and size oracle for exactly the files is_dangerous_path protects.
    """
    stub_context(KIMI_CONTEXT)
    big = _make_file(tmp_path, "big.py", KIMI_BUDGET + 5_000)

    result = check_total_file_size([big, "/etc/passwd", "relative/path.py"], "kimi")

    assert result is not None  # still rejected on the legitimate file
    reported = [entry["path"] for entry in result["metadata"]["largest_files"]]
    assert reported == [big]
    assert "/etc/passwd" not in result["content"]


def test_protected_paths_estimate_as_zero():
    """Path policy is applied before the filesystem is touched."""
    from utils.file_utils import estimate_file_tokens

    assert estimate_file_tokens("/etc/passwd") == 0
    assert estimate_file_tokens("relative/path.py") == 0


def test_no_files_passes(stub_context):
    stub_context(KIMI_CONTEXT)
    assert check_total_file_size([], "kimi") is None


def test_unresolved_model_is_rejected(tmp_path, stub_context):
    stub_context(KIMI_CONTEXT)
    f = _make_file(tmp_path, "a.py", 10)
    with pytest.raises(ValueError, match="unresolved model"):
        check_total_file_size([f], "auto")
