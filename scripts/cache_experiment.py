#!/usr/bin/env python3
"""M3 end-to-end prompt-cache experiment (`just cache-experiment`, issue #260).

Runs, against live providers, the scripted threads that docs/prompt-caching.md
section M3 defines, once per (run, arm, model), and reports the hit ratio the
provider actually served:

* **T-A** -- a 6-turn chat thread. Turn 1 attaches ``providers/gemini.py`` and
  ``utils/token_utils.py``, turn 4 attaches ``providers/openai_compatible.py``,
  all three pinned at ``288d446``. The first prompt carries a per-run nonce so a
  run cannot hit cache entries an earlier run left behind.
* **T-B** -- a consensus ``for`` / ``against`` pair on one proposal.

An **arm** is a checkout of PAL (baseline or candidate). Each (run, arm, model)
runs in a fresh worker process inside that checkout (``uv run --directory``)
with a fresh ``PAL_STATE_DIR``, so the arm's own code serves the requests. The
worker wraps the providers' ``_call_api`` and records, per provider call, the
request exactly as sent and the usage read directly from the provider's
response object -- not through PAL's parser, so the two can be cross-checked
against the transcripts. Arm order is shuffled per run.

Reported per (arm, model, scenario), with run 1 separate from runs 2-3:
median and range of the hit ratio over T-A requests 3-6 (T-B: the second consult), the hit ratio per
turn (the Stage B turn-4 criterion), SCHR predicted on the recorded bytes,
realisation (hit ratio / SCHR), sums of input / cached / written tokens,
effective input cost relative to uncached, and wall time. A request whose gap
since the previous request exceeds ``--lifetime`` is excluded.

Usage::

    just cache-experiment --arm base=../pal-base --arm cand=. --models gpt-5.5,gemini-2.5-flash --runs 3
    just cache-experiment --report results.jsonl       # re-summarise a finished run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT.parent
FIXTURE_COMMIT = "288d446"
FIXTURES_TURN_1 = ("providers/gemini.py", "utils/token_utils.py")
FIXTURES_TURN_4 = ("providers/openai_compatible.py",)
#: Requests the hit ratio is computed over: T-A requests 3-6; in T-B the second
#: consult, which shares the proposal and context file with the first.
STEADY_TURNS = {"T-A": (3, 4, 5, 6), "T-B": (2,)}
DEFAULT_READ_MULTIPLIER = 0.1
DEFAULT_WRITE_MULTIPLIER = 1.25

T_A_PROMPTS = (
    "Run {nonce}. Read the two attached files. In two sentences, what does each one do?",
    "Which function in token_utils.py would you test first, and why? Answer in two sentences.",
    "Name one edge case the Gemini provider's usage extraction could get wrong. Two sentences.",
    "I attached openai_compatible.py. In two sentences, how does its usage extraction differ from Gemini's?",
    "Which of the three files has the most duplicated logic? One sentence.",
    "Summarise this conversation in three bullet points.",
)
T_B_PROPOSAL = (
    "Proposal: cache every provider response on disk for 24 hours, keyed by the request body, "
    "so repeated PAL requests cost nothing. Evaluate it."
)


# -- pure core: usage, serialisation, summary ---------------------------------------


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(obj: Any, name: str) -> int | None:
    value = _get(obj, name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def raw_usage(raw: Any) -> dict[str, Any]:
    """Input / cached / written tokens read straight from a provider response object.

    ``cached`` is None when the provider did not report the field (unmeasured),
    except on Gemini, whose SDK omits ``cached_content_token_count`` on a miss.
    """
    metadata = _get(raw, "usage_metadata")
    if metadata is not None:
        cached = _int(metadata, "cached_content_token_count")
        return {
            "input": _int(metadata, "prompt_token_count") or 0,
            "cached": cached if cached is not None else 0,
            "written": None,
            "model": _get(raw, "model_version"),
        }
    usage = _get(raw, "usage")
    if _int(usage, "input_tokens") is not None:  # Responses API
        details = _get(usage, "input_tokens_details")
        result = {
            "input": _int(usage, "input_tokens") or 0,
            "cached": _int(details, "cached_tokens"),
            "written": _int(details, "cache_write_tokens"),
            "model": _get(raw, "model"),
        }
    else:  # Chat Completions (OpenAI, OpenRouter, llama-server)
        details = _get(usage, "prompt_tokens_details")
        result = {
            "input": _int(usage, "prompt_tokens") or 0,
            "cached": _int(details, "cached_tokens"),
            "written": _int(details, "cache_write_tokens"),
            "model": _get(raw, "model"),
        }
    upstream = _get(raw, "provider")
    if isinstance(upstream, str):
        result["upstream"] = upstream
    timings = _get(raw, "timings")
    if isinstance(timings, dict):
        result["timings"] = {k: timings[k] for k in ("cache_n", "prompt_n", "prompt_ms") if k in timings}
    return result


def _part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text", ""))
    return str(getattr(part, "text", "") or "")


def serialize_request(request: dict[str, Any]) -> str:
    """Role, a newline and the text per message, as ``cache_prefix.ser`` does for chat."""
    if "contents" in request:  # Gemini
        chunks = []
        for content in request["contents"]:
            role = _get(content, "role") or "user"
            chunks.append(f"{role}\n" + "".join(_part_text(p) for p in _get(content, "parts") or []))
        return "".join(chunks)
    params = request.get("params", {})
    messages = params.get("messages") if "messages" in params else params.get("input", [])
    chunks = []
    for message in messages or []:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(_part_text(p) for p in content)
        chunks.append(f"{message.get('role', '')}\n{content}")
    return "".join(chunks)


def _schr(texts: list[str]) -> float | None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    from cache_prefix import schr  # noqa: PLC0415  (candidate checkout only; workers never import it)

    return schr(texts)


def _ratio(records: list[dict[str, Any]]) -> float | None:
    measured = [r for r in records if r["usage"].get("cached") is not None and r["usage"]["input"]]
    if not measured:
        return None
    return sum(r["usage"]["cached"] for r in measured) / sum(r["usage"]["input"] for r in measured)


def _bucket(run: int) -> str:
    return "run 1" if run == 1 else "runs 2-3"


def summarize(
    records: list[dict[str, Any]],
    *,
    lifetime_s: float,
    read_multiplier: float = DEFAULT_READ_MULTIPLIER,
    write_multiplier: float = DEFAULT_WRITE_MULTIPLIER,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    """Group by (arm, model, scenario, run bucket) and compute the M3 numbers."""
    threads: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        threads[(record["arm"], record["model"], record["scenario"], record["run"])].append(record)

    groups: dict[tuple, list[list[dict[str, Any]]]] = defaultdict(list)
    for (arm, model, scenario, run), thread in threads.items():
        groups[(arm, model, scenario, _bucket(run))].append(sorted(thread, key=lambda r: r["turn"]))

    rows = {}
    for key, thread_list in groups.items():
        steady_ratios, schrs, per_turn = [], [], defaultdict(list)
        included: list[dict[str, Any]] = []
        excluded = 0
        wall = 0.0
        for thread in thread_list:
            wall += sum(r.get("wall_s", 0.0) for r in thread)
            schr = _schr([r["request_text"] for r in thread])
            if schr is not None:
                schrs.append(schr)
            for record in thread:
                if record["turn"] not in STEADY_TURNS.get(record["scenario"], ()):
                    continue
                if record.get("gap_s") is not None and record["gap_s"] > lifetime_s:
                    excluded += 1
                    continue
                included.append(record)
                ratio = _ratio([record])
                if ratio is not None:
                    steady_ratios.append(ratio)
                    per_turn[record["turn"]].append(ratio)
        total_input = sum(r["usage"]["input"] for r in included)
        cached = sum(r["usage"].get("cached") or 0 for r in included)
        written = sum(r["usage"].get("written") or 0 for r in included)
        median = statistics.median(steady_ratios) if steady_ratios else None
        schr_value = statistics.median(schrs) if schrs else None
        rows[key] = {
            "hit_ratio_median": median,
            "hit_ratio_range": (min(steady_ratios), max(steady_ratios)) if steady_ratios else None,
            "per_turn": {turn: statistics.median(values) for turn, values in sorted(per_turn.items())},
            "schr": schr_value,
            "realisation": median / schr_value if median is not None and schr_value else None,
            "input": total_input,
            "cached": cached,
            "written": written,
            "effective_cost": (
                ((total_input - cached - written) + cached * read_multiplier + written * write_multiplier) / total_input
                if total_input
                else None
            ),
            "excluded": excluded,
            "wall_s": wall,
            "threads": len(thread_list),
        }
    return rows


def arm_schedule(arms: list[str], *, runs: int, seed: int | None) -> list[list[str]]:
    rng = random.Random(seed)
    schedule = []
    for _ in range(runs):
        order = list(arms)
        rng.shuffle(order)
        schedule.append(order)
    return schedule


def missing_fixtures(request_text: str, names: list[str]) -> list[str]:
    """Fixture names whose contents (a ``BEGIN FILE`` block) are absent from a request."""
    return [name for name in names if not re.search(rf"--- BEGIN FILE: \S*{re.escape(name)}[ \n]", request_text)]


def default_work_root() -> Path:
    """Where run workspaces go: the harness checkout's git-ignored ``tmp/``.

    Not the system temp dir: PAL refuses to read files under /var, where macOS puts
    tempfile's directories, and under /root, a container's usual home.
    """
    return SCRIPTS_DIR.parent / "tmp" / "pal-m3"


def thread_prompts(nonce: str) -> list[str]:
    return [T_A_PROMPTS[0].format(nonce=nonce), *T_A_PROMPTS[1:]]


# -- worker: one (run, arm, model) inside the arm's checkout ------------------------


def _jsonable_request(request: dict[str, Any]) -> dict[str, Any]:
    if "contents" in request:
        return {
            "contents": [
                {"role": _get(c, "role") or "user", "parts": [{"text": _part_text(p)} for p in _get(c, "parts") or []]}
                for c in request["contents"]
            ]
        }
    params = request.get("params", {})
    kept = {k: params[k] for k in ("messages", "input") if k in params}
    return {"endpoint": request.get("endpoint"), "params": kept}


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.scenario = ""
        self.turn = 0
        self._last_start: dict[str, float] = {}

    def wrap(self, cls: type) -> None:
        original = cls._call_api
        recorder = self

        async def _call_api(provider: Any, request: dict[str, Any]) -> Any:
            start = time.time()
            raw = await original(provider, request)
            previous = recorder._last_start.get(recorder.scenario)
            recorder._last_start[recorder.scenario] = start
            jsonable = _jsonable_request(request)
            recorder.records.append(
                {
                    "scenario": recorder.scenario,
                    "turn": recorder.turn,
                    "start": start,
                    "gap_s": start - previous if previous is not None else None,
                    "wall_s": time.time() - start,
                    "request": jsonable,
                    "request_text": serialize_request(jsonable),
                    "usage": raw_usage(raw),
                }
            )
            return raw

        cls._call_api = _call_api


def _write_fixtures(directory: Path, source_repo: Path) -> dict[str, str]:
    paths = {}
    for rel in (*FIXTURES_TURN_1, *FIXTURES_TURN_4):
        content = subprocess.run(
            ["git", "-C", str(source_repo), "show", f"{FIXTURE_COMMIT}:{rel}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        target = directory / Path(rel).name
        target.write_text(content)
        paths[rel] = str(target)
    return paths


async def _run_worker(args: argparse.Namespace) -> list[dict[str, Any]]:
    arm_repo = Path.cwd()
    work_root = Path(args.work_root) if args.work_root else default_work_root()
    work_root.mkdir(parents=True, exist_ok=True)
    state_dir = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
    workspace = state_dir / "workspace"
    workspace.mkdir()
    fixtures = _write_fixtures(workspace, Path(args.fixture_repo))
    os.environ.update(
        {
            "PAL_STATE_DIR": str(state_dir),
            "PAL_LOG_DIR": str(state_dir / "logs"),
            "PAL_TRANSCRIPTS": "true",
            "LOG_LEVEL": "ERROR",
            "PAL_WORKSPACE_ROOT": str(workspace),
            "DISABLED_TOOLS": "",
        }
    )
    sys.path.insert(0, str(arm_repo))
    import logging  # noqa: PLC0415

    logging.disable(logging.CRITICAL)
    from mcp.types import CallToolRequestParams  # noqa: PLC0415

    import server  # noqa: PLC0415  (environment first)
    from providers.gemini import GeminiModelProvider  # noqa: PLC0415
    from providers.openai_compatible import OpenAICompatibleProvider  # noqa: PLC0415

    server.configure_providers()
    recorder = _Recorder()
    recorder.wrap(OpenAICompatibleProvider)
    recorder.wrap(GeminiModelProvider)

    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await server.handle_call_tool(None, CallToolRequestParams(name=tool, arguments=arguments))
        if getattr(result, "is_error", False) or getattr(result, "isError", False):
            raise RuntimeError(f"{tool} failed: {result.content}")
        return json.loads(result.content[0].text)

    recorder.scenario = "T-A"
    continuation_id = None
    for turn, prompt in enumerate(thread_prompts(args.nonce), start=1):
        recorder.turn = turn
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "model": args.model,
            "working_directory_absolute_path": str(workspace),
        }
        if turn == 1:
            arguments["absolute_file_paths"] = [fixtures[p] for p in FIXTURES_TURN_1]
        if turn == 4:
            arguments["absolute_file_paths"] = [fixtures[p] for p in FIXTURES_TURN_4]
        if continuation_id:
            arguments["continuation_id"] = continuation_id
        response = await call("chat", arguments)
        expected = [Path(p).name for p in FIXTURES_TURN_1] + (
            [Path(p).name for p in FIXTURES_TURN_4] if turn >= 4 else []
        )
        missing = missing_fixtures(recorder.records[-1]["request_text"], expected) if recorder.records else expected
        if missing:
            raise RuntimeError(
                f"T-A turn {turn}: fixture contents not in the request: {missing} (workspace {workspace})"
            )
        continuation_id = continuation_id or (response.get("continuation_offer") or {}).get("continuation_id")
        if not continuation_id:
            raise RuntimeError(f"T-A turn {turn} returned no continuation_id: {response}")

    recorder.scenario = "T-B"
    step = {
        "step": f"{T_B_PROPOSAL} (run {args.nonce})",
        "total_steps": 2,
        "findings": "Two stances on one proposal",
        "relevant_files": [fixtures[p] for p in FIXTURES_TURN_4],
        "models": [{"model": args.model, "stance": "for"}, {"model": args.model, "stance": "against"}],
    }
    recorder.turn = 1
    first = await call("consensus", {**step, "step_number": 1, "next_step_required": True})
    recorder.turn = 2
    await call(
        "consensus",
        {
            **step,
            "step": "first stance heard",
            "step_number": 2,
            "next_step_required": False,
            "continuation_id": first["continuation_offer"]["continuation_id"],
        },
    )
    for record in recorder.records:
        record.update({"arm": args.label, "run": args.run, "model": args.model, "state_dir": str(state_dir)})
    return recorder.records


# -- orchestrator ----------------------------------------------------------------


def _format(rows: dict[tuple, dict[str, Any]]) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    lines = [
        "arm | model | scenario | runs | hit 3-6 median (range) | per turn | SCHR | realisation | in/cached/written | eff. cost | wall s"
    ]
    for (arm, model, scenario, bucket), row in sorted(rows.items()):
        rng = row["hit_ratio_range"]
        per_turn = " ".join(f"{t}:{pct(v)}" for t, v in row["per_turn"].items())
        lines.append(
            f"{arm} | {model} | {scenario} | {bucket} | {pct(row['hit_ratio_median'])}"
            f" ({f'{pct(rng[0])}-{pct(rng[1])}' if rng else '-'}) | {per_turn} | {pct(row['schr'])}"
            f" | {pct(row['realisation'])} | {row['input']}/{row['cached']}/{row['written']}"
            f" | {pct(row['effective_cost'])} | {row['wall_s']:.0f}"
        )
    return "\n".join(lines)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _orchestrate(args: argparse.Namespace) -> int:
    arms = dict(spec.split("=", 1) for spec in args.arm)
    out = Path(args.out)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for run, order in enumerate(arm_schedule(list(arms), runs=args.runs, seed=args.seed), start=1):
        for label in order:
            for model in models:
                nonce = uuid.uuid4().hex[:12]
                print(f"run {run} arm {label} model {model} nonce {nonce}", file=sys.stderr)
                completed = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--directory",
                        str(Path(arms[label]).resolve()),
                        "python",
                        str(SCRIPT),
                        "--worker",
                        "--label",
                        label,
                        "--run",
                        str(run),
                        "--model",
                        model,
                        "--nonce",
                        nonce,
                        "--fixture-repo",
                        str(SCRIPTS_DIR.parent),
                        "--work-root",
                        str(Path(args.work_root).resolve() if args.work_root else default_work_root()),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    print(completed.stderr[-4000:], file=sys.stderr)
                    return completed.returncode
                with out.open("a") as handle:
                    handle.write(completed.stdout)
    print(_format(summarize(_load(out), lifetime_s=args.lifetime)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", action="append", default=[], help="label=path to a PAL checkout; repeat per arm")
    parser.add_argument("--models", default="", help="comma-separated PAL model names")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None, help="seed for the arm order")
    parser.add_argument(
        "--lifetime", type=float, default=300.0, help="cache lifetime in seconds; longer gaps are excluded"
    )
    parser.add_argument("--out", default="cache-experiment.jsonl")
    parser.add_argument("--report", help="summarise an existing results file and exit")
    parser.add_argument(
        "--work-root", help="directory for run workspaces (default <harness repo>/tmp/pal-m3; not under /var)"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    parser.add_argument("--fixture-repo", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        for record in asyncio.run(_run_worker(args)):
            print(json.dumps(record))
        return 0
    if args.report:
        print(_format(summarize(_load(Path(args.report)), lifetime_s=args.lifetime)))
        return 0
    if not args.arm or not args.models:
        parser.error("--arm and --models are required (or --report)")
    return _orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
