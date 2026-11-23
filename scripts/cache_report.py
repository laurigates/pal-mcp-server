#!/usr/bin/env python3
"""Summarise provider prompt-cache usage from PAL's conversation transcripts.

Reads ``<state dir>/threads/*.jsonl`` (see ``utils/transcripts.py``) and prints,
per (provider, model), the M2 metrics from ``docs/prompt-caching.md``:

* **coverage** -- share of calls whose usage carried ``cached_input_tokens``
  (and ``cache_write_input_tokens``). A parser records a cache key only when
  the provider reported it, so a low coverage means "unmeasured", not "missed".
* **hit ratio** -- ``sum(cached) / sum(input)`` over *steady* calls.
* **write share** -- ``sum(written) / sum(input)`` over steady calls.
* **effective cost** -- ``sum((input - cached - written) + cached*r_read +
  written*r_write) / sum(input)`` over steady calls, i.e. the input bill
  relative to the same tokens uncached. The multipliers are flags with
  generic defaults, so the number is an estimate unless they are set to the
  model's real prices.

A *call* is one assistant turn that recorded usage: ``model_metadata.usage``
on a simple-tool turn, ``model_metadata.call_usage`` on a workflow expert call
or consensus consult. Calls are numbered per thread. A call is

* ``steady`` when it is call 3 or later and used the same tool and model as
  the previous call -- the only calls the ratios are computed over;
* ``switch`` when it changed tool or model from the previous call (model
  switch or cross-tool continuation), reported in its own bucket;
* ``early`` otherwise (calls 1 and 2).

Ratios over calls missing the field use only the calls that reported it.

Usage::

    just cache-report              # text tables
    just cache-report --json       # machine-readable
    just cache-report --dir PATH   # a different transcripts directory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CACHED = "cached_input_tokens"
WRITTEN = "cache_write_input_tokens"

#: Generic multipliers, relative to the uncached input price. Reads are billed
#: at 0.05-0.25x and writes at 1.25x-2x depending on provider and model
#: (docs/prompt-caching.md), so these are placeholders to override.
DEFAULT_READ_MULTIPLIER = 0.1
DEFAULT_WRITE_MULTIPLIER = 1.25

HISTOGRAM_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-5", 4, 5),
    ("6-10", 6, 10),
    ("11+", 11, None),
)


@dataclass(frozen=True)
class Call:
    """One model call that recorded usage."""

    thread_id: str
    index: int
    tool_name: str | None
    provider: str
    model: str
    usage: dict[str, int]
    bucket: str


@dataclass
class Totals:
    """Token sums over a set of calls."""

    calls: int = 0
    input_tokens: int = 0
    cached_calls: int = 0
    cached_input: int = 0
    cached_tokens: int = 0
    written_calls: int = 0
    written_input: int = 0
    written_tokens: int = 0
    #: Written tokens on the calls that reported the cached field (0 where a
    #: call reported no write field), so effective cost sums one set of calls.
    written_in_cached: int = 0

    def add(self, usage: dict[str, int]) -> None:
        self.calls += 1
        input_tokens = _int(usage.get("input_tokens")) or 0
        self.input_tokens += input_tokens
        cached = _int(usage.get(CACHED))
        if cached is not None:
            self.cached_calls += 1
            self.cached_input += input_tokens
            self.cached_tokens += cached
            self.written_in_cached += _int(usage.get(WRITTEN)) or 0
        written = _int(usage.get(WRITTEN))
        if written is not None:
            self.written_calls += 1
            self.written_input += input_tokens
            self.written_tokens += written

    def hit_ratio(self) -> float | None:
        return _ratio(self.cached_tokens, self.cached_input)

    def write_share(self) -> float | None:
        return _ratio(self.written_tokens, self.written_input)

    def effective_cost(self, read_multiplier: float, write_multiplier: float) -> float | None:
        """Input bill relative to uncached, over the calls that reported the cached field."""
        if not self.cached_input:
            return None
        written = self.written_in_cached
        uncached = self.cached_input - self.cached_tokens - written
        billed = uncached + self.cached_tokens * read_multiplier + written * write_multiplier
        return billed / self.cached_input


@dataclass
class ModelReport:
    """Everything reported for one (provider, model)."""

    provider: str
    model: str
    all_calls: Totals = field(default_factory=Totals)
    buckets: dict[str, Totals] = field(default_factory=lambda: defaultdict(Totals))


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _call_fields(turn: dict[str, Any]) -> tuple[str | None, str | None, dict[str, int]] | None:
    """Return (model, provider, usage) for an assistant turn that recorded usage."""
    metadata = turn.get("model_metadata")
    if not isinstance(metadata, dict):
        return None
    call_usage = metadata.get("call_usage")
    if isinstance(call_usage, dict) and isinstance(call_usage.get("usage"), dict):
        return call_usage.get("model_name"), call_usage.get("provider"), call_usage["usage"]
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        return turn.get("model_name"), turn.get("model_provider"), usage
    return None


def thread_calls(thread_id: str, turns: Iterable[dict[str, Any]]) -> list[Call]:
    """Number and classify the calls of one thread, in turn order."""
    calls: list[Call] = []
    previous: tuple[str | None, str] | None = None
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        fields = _call_fields(turn)
        if fields is None:
            continue
        model, provider, usage = fields
        model = model or "unknown"
        tool_name = turn.get("tool_name")
        index = len(calls) + 1
        current = (tool_name, model)
        if previous is not None and current != previous:
            bucket = "switch"
        elif index >= 3:
            bucket = "steady"
        else:
            bucket = "early"
        calls.append(Call(thread_id, index, tool_name, provider or "unknown", model, usage, bucket))
        previous = current
    return calls


def read_thread_turns(path: Path) -> list[dict[str, Any]]:
    """Return the turn records of one transcript; malformed lines are skipped."""
    turns: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("type") == "turn":
                turns.append(record)
    return turns


def histogram(calls_per_thread: Iterable[int]) -> dict[str, int]:
    """Bucket the number of calls per thread."""
    counts = dict.fromkeys((label for label, _, _ in HISTOGRAM_BUCKETS), 0)
    for count in calls_per_thread:
        for label, low, high in HISTOGRAM_BUCKETS:
            if count >= low and (high is None or count <= high):
                counts[label] += 1
                break
    return counts


def build_report(
    threads: dict[str, list[dict[str, Any]]],
    read_multiplier: float = DEFAULT_READ_MULTIPLIER,
    write_multiplier: float = DEFAULT_WRITE_MULTIPLIER,
) -> dict[str, Any]:
    """Compute the report for ``{thread_id: turn records}``."""
    models: dict[tuple[str, str], ModelReport] = {}
    calls_per_thread: list[int] = []
    bucket_counts: Counter[str] = Counter()
    for thread_id, turns in threads.items():
        calls = thread_calls(thread_id, turns)
        if not calls:
            continue
        calls_per_thread.append(len(calls))
        for call in calls:
            key = (call.provider, call.model)
            report = models.setdefault(key, ModelReport(call.provider, call.model))
            report.all_calls.add(call.usage)
            report.buckets[call.bucket].add(call.usage)
            bucket_counts[call.bucket] += 1

    rows = []
    for (provider, model), report in sorted(models.items()):
        steady = report.buckets.get("steady", Totals())
        switch = report.buckets.get("switch", Totals())
        rows.append(
            {
                "provider": provider,
                "model": model,
                "calls": report.all_calls.calls,
                "cached_field_coverage": _ratio(report.all_calls.cached_calls, report.all_calls.calls),
                "write_field_coverage": _ratio(report.all_calls.written_calls, report.all_calls.calls),
                "steady_calls": steady.calls,
                "hit_ratio": steady.hit_ratio(),
                "write_share": steady.write_share(),
                "effective_cost": steady.effective_cost(read_multiplier, write_multiplier),
                "input_tokens": steady.input_tokens,
                "cached_tokens": steady.cached_tokens,
                "written_tokens": steady.written_tokens,
                "switch_calls": switch.calls,
                "switch_hit_ratio": switch.hit_ratio(),
            }
        )
    return {
        "threads": len(calls_per_thread),
        "calls_by_bucket": {name: bucket_counts.get(name, 0) for name in ("early", "steady", "switch")},
        "calls_per_thread_histogram": histogram(calls_per_thread),
        "multipliers": {"read": read_multiplier, "write": write_multiplier},
        "models": rows,
    }


def load_threads(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Read every transcript in ``directory``; unreadable files are skipped."""
    threads: dict[str, list[dict[str, Any]]] = {}
    if not directory.is_dir():
        return threads
    for path in sorted(directory.glob("*.jsonl")):
        try:
            threads[path.stem] = read_thread_turns(path)
        except OSError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
    return threads


def _fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_text(report: dict[str, Any], directory: Path) -> str:
    """Render the report as plain-text tables."""
    lines = [f"Prompt-cache report for {directory}", ""]
    if not report["models"]:
        lines.append("No calls with recorded usage found.")
        return "\n".join(lines)

    buckets = report["calls_by_bucket"]
    lines.append(
        f"{report['threads']} threads; calls: {buckets['early']} early, "
        f"{buckets['steady']} steady, {buckets['switch']} model-switch/cross-tool"
    )
    multipliers = report["multipliers"]
    lines.append(
        f"Effective cost uses read x{multipliers['read']} and write x{multipliers['write']} "
        "(estimate unless set to the model's prices)"
    )
    lines.append("")
    header = (
        "provider",
        "model",
        "calls",
        "cov",
        "wcov",
        "steady",
        "hit",
        "write",
        "eff.cost",
        "switch",
        "sw.hit",
    )
    table = [header]
    for row in report["models"]:
        table.append(
            (
                row["provider"],
                row["model"],
                str(row["calls"]),
                _fmt_ratio(row["cached_field_coverage"]),
                _fmt_ratio(row["write_field_coverage"]),
                str(row["steady_calls"]),
                _fmt_ratio(row["hit_ratio"]),
                _fmt_ratio(row["write_share"]),
                _fmt_ratio(row["effective_cost"]),
                str(row["switch_calls"]),
                _fmt_ratio(row["switch_hit_ratio"]),
            )
        )
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    for r in table:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(r, widths, strict=True)).rstrip())
    lines.append("")
    lines.append("Calls per thread:")
    for label, count in report["calls_per_thread_histogram"].items():
        lines.append(f"  {label:>5}  {count}")
    return "\n".join(lines)


def _default_directory() -> Path:
    sys.path.insert(0, os.getcwd())
    from utils.transcripts import get_transcripts_dir  # noqa: PLC0415  (repo root on sys.path first)

    return get_transcripts_dir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, help="transcripts directory (default: <state dir>/threads)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text tables")
    parser.add_argument("--read-multiplier", type=float, default=DEFAULT_READ_MULTIPLIER)
    parser.add_argument("--write-multiplier", type=float, default=DEFAULT_WRITE_MULTIPLIER)
    args = parser.parse_args(argv)

    directory = args.dir or _default_directory()
    report = build_report(load_threads(directory), args.read_multiplier, args.write_multiplier)
    if args.json:
        print(json.dumps({"directory": str(directory), **report}, indent=2))
    else:
        print(render_text(report, directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
