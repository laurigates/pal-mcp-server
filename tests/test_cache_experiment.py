"""Pure parts of the M3 harness (scripts/cache_experiment.py, docs/prompt-caching.md M3).

The live run needs provider keys, so what is pinned here is what turns the
captured raw requests and responses into the reported numbers: reading usage
straight from the provider's own response objects (independent of PAL's
parser), serialising each request for SCHR, and the per-arm summary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import cache_experiment as ce  # noqa: E402


class TestRawUsage:
    def test_chat_completions_with_cached_and_written(self):
        raw = NS(
            model="gpt-5.5-2026-01-01",
            usage=NS(prompt_tokens=9000, prompt_tokens_details=NS(cached_tokens=6144, cache_write_tokens=512)),
        )
        assert ce.raw_usage(raw) == {"input": 9000, "cached": 6144, "written": 512, "model": "gpt-5.5-2026-01-01"}

    def test_chat_completions_miss_without_details_is_unmeasured_not_zero(self):
        raw = NS(model="m", usage=NS(prompt_tokens=100, prompt_tokens_details=None))
        assert ce.raw_usage(raw) == {"input": 100, "cached": None, "written": None, "model": "m"}

    def test_openrouter_upstream_provider_is_recorded(self):
        raw = NS(model="m", provider="DeepInfra", usage=NS(prompt_tokens=10, prompt_tokens_details=NS(cached_tokens=0)))
        assert ce.raw_usage(raw)["upstream"] == "DeepInfra"

    def test_responses_api(self):
        raw = NS(model="gpt-6", usage=NS(input_tokens=5000, input_tokens_details=NS(cached_tokens=4096)))
        assert ce.raw_usage(raw) == {"input": 5000, "cached": 4096, "written": None, "model": "gpt-6"}

    def test_gemini_miss_reports_absent_cached_field_as_zero(self):
        """The SDK omits cached_content_token_count on a miss (Stage 0 control calls)."""
        raw = NS(model_version="gemini-2.5-flash", usage_metadata=NS(prompt_token_count=8182))
        assert ce.raw_usage(raw) == {"input": 8182, "cached": 0, "written": None, "model": "gemini-2.5-flash"}

    def test_gemini_hit(self):
        raw = NS(
            model_version="gemini-2.5-flash",
            usage_metadata=NS(prompt_token_count=8182, cached_content_token_count=8170),
        )
        assert ce.raw_usage(raw)["cached"] == 8170

    def test_llama_server_timings_are_kept(self):
        raw = NS(
            model="local",
            usage=NS(prompt_tokens=7000, prompt_tokens_details=NS(cached_tokens=6500)),
            timings={"cache_n": 6500, "prompt_n": 500, "prompt_ms": 120.0},
        )
        assert ce.raw_usage(raw)["timings"] == {"cache_n": 6500, "prompt_n": 500, "prompt_ms": 120.0}


class TestSerialize:
    def test_chat_messages(self):
        request = {
            "endpoint": "chat",
            "params": {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]},
        }
        assert ce.serialize_request(request) == "system\nSuser\nU"

    def test_responses_input(self):
        request = {
            "endpoint": "responses",
            "params": {"input": [{"role": "user", "content": [{"type": "input_text", "text": "U"}]}]},
        }
        assert ce.serialize_request(request) == "user\nU"

    def test_gemini_contents(self):
        request = {"contents": [{"parts": [{"text": "SYS\n\nU"}]}]}
        assert ce.serialize_request(request) == "user\nSYS\n\nU"


def _record(arm, run, turn, input_, cached, text, gap=1.0, written=None, scenario="T-A", model="gpt-5.5"):
    return {
        "arm": arm,
        "run": run,
        "model": model,
        "scenario": scenario,
        "turn": turn,
        "gap_s": gap,
        "wall_s": 2.0,
        "request_text": text,
        "usage": {"input": input_, "cached": cached, "written": written},
    }


def _thread(arm, run, cached_by_turn, input_=10_000):
    text = "x" * 40_000
    return [_record(arm, run, t, input_, c, text + str(t) * t) for t, c in enumerate(cached_by_turn, start=1)]


class TestSummarize:
    def test_median_and_range_over_requests_3_to_6(self):
        records = _thread("cand", 2, [0, 5000, 6000, 8000, 7000, 9000])
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-A", "runs 2-3")]
        assert row["hit_ratio_median"] == pytest.approx(0.75)
        assert row["hit_ratio_range"] == (pytest.approx(0.6), pytest.approx(0.9))

    def test_run_1_is_reported_separately(self):
        records = _thread("cand", 1, [0, 0, 0, 0, 0, 0]) + _thread("cand", 2, [0, 0, 9000, 9000, 9000, 9000])
        rows = ce.summarize(records, lifetime_s=300)
        assert rows[("cand", "gpt-5.5", "T-A", "run 1")]["hit_ratio_median"] == 0
        assert rows[("cand", "gpt-5.5", "T-A", "runs 2-3")]["hit_ratio_median"] == pytest.approx(0.9)

    def test_requests_past_the_cache_lifetime_are_excluded(self):
        records = _thread("cand", 2, [0, 0, 9000, 9000, 0, 9000])
        records[4]["gap_s"] = 900
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-A", "runs 2-3")]
        assert row["excluded"] == 1
        assert row["hit_ratio_median"] == pytest.approx(0.9)

    def test_per_turn_hit_ratio_for_the_turn_4_dip(self):
        records = _thread("cand", 2, [0, 0, 8000, 2000, 8000, 8000])
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-A", "runs 2-3")]
        assert row["per_turn"][4] == pytest.approx(0.2)

    def test_effective_cost_uses_read_and_write_multipliers(self):
        records = _thread("cand", 2, [0, 0, 5000, 5000, 5000, 5000])
        records[2]["usage"]["written"] = 1000
        row = ce.summarize(records, lifetime_s=300, read_multiplier=0.1, write_multiplier=1.25)[
            ("cand", "gpt-5.5", "T-A", "runs 2-3")
        ]
        # turns 3-6: 40,000 input; 20,000 cached at 0.1; 1,000 written at 1.25; 19,000 uncached
        assert row["effective_cost"] == pytest.approx((19_000 + 2_000 + 1_250) / 40_000)

    def test_realisation_is_hit_ratio_over_schr(self):
        records = _thread("cand", 2, [0, 0, 5000, 5000, 5000, 5000])
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-A", "runs 2-3")]
        assert row["schr"] is not None and row["schr"] > 0
        assert row["realisation"] == pytest.approx(row["hit_ratio_median"] / row["schr"])

    def test_unmeasured_cached_field_yields_no_ratio(self):
        records = _thread("cand", 2, [None] * 6)
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-A", "runs 2-3")]
        assert row["hit_ratio_median"] is None

    def test_consensus_pair_reports_the_second_consult(self):
        records = [
            _record("cand", 2, 1, 10_000, 0, "a" * 40_000, scenario="T-B"),
            _record("cand", 2, 2, 10_000, 7000, "a" * 40_000, scenario="T-B"),
        ]
        row = ce.summarize(records, lifetime_s=300)[("cand", "gpt-5.5", "T-B", "runs 2-3")]
        assert row["hit_ratio_median"] == pytest.approx(0.7)


class TestArmOrder:
    def test_every_run_has_every_arm_once_and_order_varies_with_seed(self):
        schedules = {tuple(tuple(r) for r in ce.arm_schedule(["base", "cand"], runs=3, seed=s)) for s in range(20)}
        for schedule in schedules:
            assert all(sorted(run) == ["base", "cand"] for run in schedule)
        assert len(schedules) > 1


class TestNonce:
    def test_first_prompt_carries_a_per_run_nonce(self):
        prompts = ce.thread_prompts(nonce="abc123")
        assert "abc123" in prompts[0]
        assert all("abc123" not in p for p in prompts[1:])
        assert len(prompts) == 6


class TestFixturesEmbedded:
    """A file PAL refuses to read (macOS /var temp dirs) must stop the run, not shrink it (#260)."""

    def test_missing_contents_are_reported(self):
        text = "Error: Access to system directory denied: /var/folders/x/gemini.py"
        assert ce.missing_fixtures(text, ["gemini.py"]) == ["gemini.py"]

    def test_embedded_file_passes(self):
        text = "--- BEGIN FILE: /home/u/m3/gemini.py ---\ncode\n--- END FILE: /home/u/m3/gemini.py ---"
        assert ce.missing_fixtures(text, ["gemini.py"]) == []

    def test_default_work_root_is_outside_system_directories(self):
        from utils.security_config import DANGEROUS_SYSTEM_PATHS

        root = str(ce.default_work_root())
        assert not any(root == d or root.startswith(d.rstrip("/") + "/") for d in DANGEROUS_SYSTEM_PATHS if d != "/")
