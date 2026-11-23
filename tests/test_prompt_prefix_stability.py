"""Prompt-prefix gates G0-G7 from docs/prompt-caching.md (M1), via `just cache-prefix`.

``scripts/cache_prefix.py --json`` runs once per module in a fresh interpreter
(it imports ``server`` with its own environment and spawns more interpreters for
G4). Every gate fails on today's code, for the reasons the design document
gives (#187's double-recorded prompt, the ``Turn N/50`` header, mtime and
newest-first file ordering, set iteration in expert prompts, the consensus
stance inside the system prompt), so each starts as ``xfail(strict=True)``. The
PR that makes a gate pass removes its marker; a gate that starts passing
without that PR fails the suite, which is the point of ``strict``.

The gate functions are also tested on synthetic requests, so each gate is known
to be passable by a correct request sequence and not only failable.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "cache_prefix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cache_prefix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cache_prefix"] = module
    spec.loader.exec_module(module)
    return module


cache_prefix = _load_module()


@pytest.fixture(scope="module")
def report() -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout)


# -- the harness itself ------------------------------------------------------------


def test_all_gates_are_reported(report):
    assert sorted(report["gates"]) == ["G0", "G1", "G2", "G3", "G4", "G5", "G7"]


def test_scenarios_record_every_request(report):
    scenarios = report["scenarios"]
    assert scenarios["S1"]["requests"] == 4
    assert scenarios["S2"]["requests"] == 4
    assert scenarios["S3"]["requests"] == 5
    assert scenarios["S5"]["requests"] == 12
    assert scenarios["S6"]["requests"] == 2
    assert scenarios["S7"]["requests"] == 3


def test_s5_overflows_the_history_budget(report):
    """Without an overflow, G7 would measure nothing."""
    assert report["scenarios"]["S5"]["history_overflowed"] is True


def test_g4_control_runs_agree(report):
    """The same seed twice must give the same bytes, or G4's comparison means nothing."""
    assert "harness_error" not in report["gates"]["G4"]


# -- gates (switched on by the stage that makes each pass) -------------------------


@pytest.mark.xfail(strict=True, reason="#187 records the prompt twice; Stage A (#194)")
def test_g0_completeness(report):
    assert report["gates"]["G0"]["passed"], report["gates"]["G0"]["problems"]


@pytest.mark.xfail(strict=True, reason="Turn N/50 header is the first varying byte; Stage A (#194)")
def test_g1_chat_prefix_reaches_past_last_turn(report):
    assert report["gates"]["G1"]["passed"], report["gates"]["G1"]["problems"]


@pytest.mark.xfail(strict=True, reason="newest-first files with mtime headers; Stage B (#195)")
def test_g2_new_files_keep_the_prefix(report):
    assert report["gates"]["G2"]["passed"], report["gates"]["G2"]["problems"]


@pytest.mark.xfail(strict=True, reason="volatile trailer; Stage C (#196)")
def test_g3_requests_are_append_only(report):
    assert report["gates"]["G3"]["passed"], report["gates"]["G3"]["problems"]


@pytest.mark.xfail(strict=True, reason="expert prompts iterate sets; Stage A (#194)")
def test_g4_expert_prompts_ignore_hash_seed(report):
    assert report["gates"]["G4"]["passed"], report["gates"]["G4"]["problems"]


@pytest.mark.xfail(strict=True, reason="stance spliced into the system prompt; Stage A (#194)")
def test_g5_consensus_consults_differ_only_at_the_stance(report):
    assert report["gates"]["G5"]["passed"], report["gates"]["G5"]["problems"]


@pytest.mark.xfail(strict=True, reason="history window slides every call after overflow; Stage B (#195)")
def test_g7_history_compaction(report):
    assert report["gates"]["G7"]["passed"], report["gates"]["G7"]["problems"]


# -- gate functions on synthetic requests -----------------------------------------


def _continuation(turns: list[tuple[str, str]], new_input: str) -> str:
    blocks = "".join(
        f"\n--- Turn {i} (Agent) ---\n{p}\n--- Turn {i} (stub) ---\nREPLY" for i, (p, _) in enumerate(turns, 1)
    )
    return f"system\nSYS\nuser\n=== HISTORY ==={blocks}\n\n=== END CONVERSATION HISTORY ===\n{new_input}"


def test_append_only_sequence_passes_g1_and_g3():
    texts = ["system\nSYS\nuser\nfirst"]
    for k in range(2, 5):
        texts.append(texts[-1] + f" more{k}")
    assert cache_prefix.gate_append_only(texts) == []
    assert cache_prefix.gate_reaches_past_last_turn(cache_prefix._pairs(texts)) == []


def test_history_sequence_with_stable_prefix_passes_g1():
    texts = [_continuation([("p1", "r")] * k, f"input {k}") for k in range(1, 4)]
    assert cache_prefix.gate_reaches_past_last_turn(cache_prefix._pairs(texts)) == []


def test_a_varying_header_fails_g1():
    texts = [_continuation([("p1", "r")] * k, "x").replace("=== HISTORY ===", f"Turn {k}/50") for k in range(1, 4)]
    assert cache_prefix.gate_reaches_past_last_turn(cache_prefix._pairs(texts))


def test_completeness_counts_prompts_replies_and_files():
    prompts = ["P-one", "P-two"]
    good = ["P-one SENT-A", "P-one REPLY P-two SENT-A"]
    assert cache_prefix.gate_completeness(good, prompts, {1: ["SENT-A"]}, "REPLY") == []

    doubled = ["P-one SENT-A", "P-one REPLY P-two P-two SENT-A SENT-A"]
    problems = cache_prefix.gate_completeness(doubled, prompts, {1: ["SENT-A"]}, "REPLY")
    assert any("user prompt 2 appears 2x" in p for p in problems)
    assert any("SENT-A appears 2x" in p for p in problems)


def _consult(file_body: str, stance: str) -> str:
    end = cache_prefix.FIXTURE_ENDS["alpha.py"]
    return f"system\nSYS\nuser\nPROPOSAL\n# SENTINEL-ALPHA\n{file_body}\n# {end}\n\nSTANCE: {stance}"


def test_g5_passes_consults_that_differ_only_at_the_stance():
    consults = [_consult("body", "for"), _consult("body", "against")]
    assert cache_prefix.gate_consults_share_context(consults, "PROPOSAL", "alpha.py") == []


def test_g5_fails_consults_that_differ_inside_the_context_file():
    """#248: sharing up to the start of the file is not enough; the whole file must be shared."""
    consults = [_consult("body one", "for"), _consult("body two", "against")]
    assert cache_prefix.gate_consults_share_context(consults, "PROPOSAL", "alpha.py")


def test_g5_fails_when_the_context_file_is_missing():
    consults = ["system\nSYS\nuser\nPROPOSAL for", "system\nSYS\nuser\nPROPOSAL against"]
    assert cache_prefix.gate_consults_share_context(consults, "PROPOSAL", "alpha.py")


def test_fixture_end_markers_do_not_contain_the_start_sentinels():
    """G0 counts start sentinels; an end marker containing one would double the count."""
    for name, end in cache_prefix.FIXTURE_ENDS.items():
        assert all(sentinel not in end for sentinel in cache_prefix.FIXTURES.values()), name


def test_ser_hashes_image_parts():
    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
            }
        ]
    }
    text = cache_prefix.ser(request)
    assert text.startswith("user\nlook<img:") and len(text) == len("user\nlook<img:") + 16 + 1
