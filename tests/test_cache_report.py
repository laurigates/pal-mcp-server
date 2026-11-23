"""Tests for scripts/cache_report.py (`just cache-report`, issue #193).

The report reads transcripts as utils/transcripts.py writes them, so the
fixture threads below are built from the same turn shapes: simple tools store
``model_metadata.usage``, workflow expert calls and consensus consults store
``model_metadata.call_usage``. One end-to-end test writes a real transcript
through ``record_turn`` so a change to the stored shape breaks here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "cache_report.py"


def _load_module():
    """Import the script by path -- scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("cache_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cache_report"] = module
    spec.loader.exec_module(module)
    return module


cache_report = _load_module()


def _simple(model: str, usage: dict, tool: str = "chat", provider: str = "google") -> list[dict]:
    """A user turn plus a simple-tool assistant turn."""
    return [
        {"type": "turn", "role": "user", "content": "q", "tool_name": tool},
        {
            "type": "turn",
            "role": "assistant",
            "content": "a",
            "tool_name": tool,
            "model_name": model,
            "model_provider": provider,
            "model_metadata": {"usage": usage, "metadata": {}},
        },
    ]


def _row(report: dict, model: str) -> dict:
    (row,) = [r for r in report["models"] if r["model"] == model]
    return row


def test_steady_calls_start_at_call_three_with_the_same_tool_and_model():
    turns = (
        _simple("flash", {"input_tokens": 1000, "cached_input_tokens": 0})
        + _simple("flash", {"input_tokens": 1000, "cached_input_tokens": 0})
        + _simple("flash", {"input_tokens": 2000, "cached_input_tokens": 1500})
        + _simple("flash", {"input_tokens": 2000, "cached_input_tokens": 1700})
    )

    calls = cache_report.thread_calls("t1", turns)

    assert [c.bucket for c in calls] == ["early", "early", "steady", "steady"]
    row = _row(cache_report.build_report({"t1": turns}), "flash")
    assert row["steady_calls"] == 2
    assert row["hit_ratio"] == pytest.approx(3200 / 4000)


def test_model_switch_and_cross_tool_calls_get_their_own_bucket():
    turns = (
        _simple("flash", {"input_tokens": 100})
        + _simple("flash", {"input_tokens": 100})
        + _simple("o3", {"input_tokens": 100, "cached_input_tokens": 50}, provider="openai")
        + _simple("o3", {"input_tokens": 100}, tool="debug", provider="openai")
        + _simple("o3", {"input_tokens": 100}, tool="debug", provider="openai")
    )

    buckets = [c.bucket for c in cache_report.thread_calls("t1", turns)]

    assert buckets == ["early", "early", "switch", "switch", "steady"]
    row = _row(cache_report.build_report({"t1": turns}), "o3")
    assert row["switch_calls"] == 2
    assert row["switch_hit_ratio"] == pytest.approx(0.5)


def test_workflow_and_consensus_turns_are_read_from_call_usage():
    turns = [
        {"type": "turn", "role": "user", "content": "step", "tool_name": "consensus"},
        {
            "type": "turn",
            "role": "assistant",
            "content": "consulted",
            "tool_name": "consensus",
            "model_name": None,
            "model_provider": None,
            "model_metadata": {
                "work_history": [],
                "call_usage": {"model_name": "o3", "provider": "openai", "usage": {"input_tokens": 7}},
            },
        },
        # A workflow step that called no model stores state but no usage.
        {
            "type": "turn",
            "role": "assistant",
            "content": "paused",
            "tool_name": "consensus",
            "model_metadata": {"work_history": []},
        },
    ]

    calls = cache_report.thread_calls("t1", turns)

    assert [(c.provider, c.model, c.usage) for c in calls] == [("openai", "o3", {"input_tokens": 7})]


def test_coverage_counts_absence_and_ratios_use_only_reporting_calls():
    turns = (
        _simple("m", {"input_tokens": 1000})
        + _simple("m", {"input_tokens": 1000})
        + _simple("m", {"input_tokens": 1000, "cached_input_tokens": 600, "cache_write_input_tokens": 100})
        + _simple("m", {"input_tokens": 5000})  # no cache field: excluded from the ratio
    )

    row = _row(cache_report.build_report({"t1": turns}, read_multiplier=0.1, write_multiplier=1.25), "m")

    assert row["cached_field_coverage"] == pytest.approx(0.25)
    assert row["write_field_coverage"] == pytest.approx(0.25)
    assert row["hit_ratio"] == pytest.approx(0.6)
    assert row["write_share"] == pytest.approx(0.1)
    # (1000 - 600 - 100) + 600 * 0.1 + 100 * 1.25 = 485 of 1000 uncached
    assert row["effective_cost"] == pytest.approx(0.485)


def test_histogram_counts_calls_per_thread():
    threads = {
        "a": _simple("m", {"input_tokens": 1}),
        "b": _simple("m", {"input_tokens": 1}) * 2,
        "c": _simple("m", {"input_tokens": 1}) * 7,
        "empty": [{"type": "turn", "role": "user", "content": "q"}],
    }

    report = cache_report.build_report(threads)

    assert report["threads"] == 3
    assert report["calls_per_thread_histogram"] == {"1": 1, "2": 1, "3": 0, "4-5": 0, "6-10": 1, "11+": 0}


def test_reads_transcripts_written_by_the_server(tmp_path, monkeypatch, capsys):
    """Round trip through utils.transcripts so the report tracks the stored shape."""
    from utils.conversation_memory import ConversationTurn
    from utils.transcripts import record_turn

    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path))
    thread_id = "11111111-1111-4111-8111-111111111111"
    for index in range(3):
        record_turn(thread_id, ConversationTurn(role="user", content="q", timestamp="t", tool_name="chat"))
        record_turn(
            thread_id,
            ConversationTurn(
                role="assistant",
                content="a",
                timestamp="t",
                tool_name="chat",
                model_name="flash",
                model_provider="google",
                model_metadata={"usage": {"input_tokens": 1000, "cached_input_tokens": 400 * index}},
            ),
        )

    assert cache_report.main(["--dir", str(tmp_path / "threads"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    row = _row(report, "flash")
    assert row["calls"] == 3
    assert row["steady_calls"] == 1
    assert row["hit_ratio"] == pytest.approx(0.8)


def test_text_output_without_transcripts(tmp_path, capsys):
    assert cache_report.main(["--dir", str(tmp_path / "missing")]) == 0
    assert "No calls with recorded usage found." in capsys.readouterr().out
