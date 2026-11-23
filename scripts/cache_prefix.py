#!/usr/bin/env python3
"""Offline prompt-prefix harness and gates G0-G7 (`just cache-prefix`, issue #193).

Runs the strict stub (``simulator_tests/stub_provider.py``) in this process,
drives scripted scenarios through ``server.handle_call_tool`` -- so ``server.py``
and ``SimpleTool`` are covered, where #187 lives -- and records every request
body exactly as the stub received it. The scenarios, the serialisation
``ser(r)``, ``L_j = LCP(ser(r_j), ser(r_{j+1}))`` and the gates are defined in
docs/prompt-caching.md, section M1.

Scenarios: S1 chat without files; S2 the same fixture every turn; S3 a new
fixture at turns 1, 3 and 5; S5 a small-context model over 12 turns; S6
consensus ``for`` then ``against``; S7 codereview, debug and thinkdeep expert
requests with two relevant files. (S4, a fixture edited between turns, is
informational in the design and not implemented here.)

Gates are structural, so they cannot be passed by sending less:

* G0 every request carries each user prompt so far exactly once, each earlier
  stub reply, and each attached file exactly once (S1-S3).
* G1 / G2 ``L_j`` reaches past the end of the last turn block of ``r_j``
  (S1-S2 / S3).
* G3 ``ser(r_{j+1})`` starts with ``ser(r_j)`` (S1-S3).
* G4 S7's request bodies hash identically under PYTHONHASHSEED 1, 2 and 3,
  after a same-seed control run (seed 1 twice) agrees with itself. A control
  that disagrees is a harness error, not a gate result.
* G5 the S6 ``for`` and ``against`` consults share everything up to and
  including the proposal and the last line of its context file.
* G7 at most 2 of S5's pairs have ``L_j`` before the last turn block of ``r_j``.

Also reported, not gated: the simulated cache-hit ratio
``SCHR = sum(cacheable(L_j)) / sum(tokens(r_{j+1}))`` with ``tokens(c) = c // 4``
and ``cacheable`` counting a prefix only from 1,024 tokens (OpenAI's documented
minimum). It is an upper-bound proxy, not a prediction of ``cached_tokens``.

Usage::

    just cache-prefix            # summary
    just cache-prefix --json     # machine-readable; tests/test_prompt_prefix_stability.py reads it
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY = REPO_ROOT / "simulator_tests" / "conf" / "cache_prefix_models.json"

MIN_CACHEABLE_TOKENS = 1024
G7_MAX_BROKEN_PAIRS = 2
HASH_SEEDS = ("1", "2", "3")
END_HISTORY = "=== END CONVERSATION HISTORY ==="
TURN_HEADER = "\n--- Turn "
HISTORY_TRUNCATED = "[Note: Showing "

#: Provider keys scrubbed so the Custom provider (the stub) is the only one configured.
PROVIDER_KEYS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DIAL_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENCODE_API_KEY",
    "OPENCODE_GO_API_KEY",
)

# Each fixture carries a sentinel line, so G0 can count the file's presence in a
# request regardless of how PAL decorates the content (line numbers, headers).
FIXTURES = {
    "alpha.py": "SENTINEL-ALPHA",
    "beta.py": "SENTINEL-BETA",
    "gamma.py": "SENTINEL-GAMMA",
}
# ...and a last line, so G5 can locate where the file's content ends (#248). The
# end markers must not contain a start sentinel, or G0 would count it twice.
FIXTURE_ENDS = {name: f"FIXTURE-END-{name}" for name in FIXTURES}


# -- serialisation and measures -------------------------------------------------


def ser(request: dict[str, Any]) -> str:
    """Per message: role, a newline and the text; image parts become ``<img:sha256[:16]>``."""
    parts = []
    for message in request.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url", ""))
                    texts.append(f"<img:{hashlib.sha256(url.encode()).hexdigest()[:16]}>")
                else:
                    texts.append(str(part.get("text", "")))
            content = "".join(texts)
        parts.append(f"{message.get('role', '')}\n{content}")
    return "".join(parts)


def lcp(a: str, b: str) -> int:
    return len(os.path.commonprefix([a, b]))


def end_of_last_turn_block(text: str) -> int | None:
    """Index where the last rendered turn block of a continuation ends, or None without history."""
    last_turn = text.rfind(TURN_HEADER)
    if last_turn < 0:
        return None
    end = text.find(END_HISTORY, last_turn)
    if end < 0:
        return None
    # The block ends at its last non-blank character: the separator before the
    # END marker is where the next turn block goes, so it is not shared.
    return len(text[:end].rstrip())


def schr(texts: list[str]) -> float | None:
    cached = 0
    total = 0
    for previous, current in zip(texts, texts[1:], strict=False):
        prefix_tokens = lcp(previous, current) // 4
        cached += prefix_tokens if prefix_tokens >= MIN_CACHEABLE_TOKENS else 0
        total += len(current) // 4
    return cached / total if total else None


# -- environment and server -------------------------------------------------------


def _prepare_environment(base_url: str, state_dir: Path, workspace: Path) -> None:
    """Point the server at the stub and nothing else; must run before importing it."""
    for key in PROVIDER_KEYS:
        os.environ.pop(key, None)
    os.environ.update(
        {
            "CUSTOM_API_URL": base_url,
            "CUSTOM_MODELS_CONFIG_PATH": str(REGISTRY),
            "DEFAULT_MODEL": "ci-local",
            "PAL_STATE_DIR": str(state_dir),
            "PAL_LOG_DIR": str(state_dir / "logs"),
            "PAL_TRANSCRIPTS": "false",
            "LOG_LEVEL": "ERROR",
            "LOCALE": "",
            "DISABLED_TOOLS": "",
            "PAL_WORKSPACE_ROOT": str(workspace),
        }
    )
    sys.path.insert(0, str(REPO_ROOT))


def _text(result: Any) -> dict[str, Any]:
    if getattr(result, "is_error", False):
        raise RuntimeError(f"tool call failed: {result.content}")
    return json.loads(result.content[0].text)


def _load_stub():
    """Import the stub by path: the simulator_tests package imports server modules on load,
    which must not happen before _prepare_environment has run."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("stub_provider", REPO_ROOT / "simulator_tests" / "stub_provider.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Harness:
    """Drives scenarios against one in-process stub and records request bodies."""

    def __init__(self, fixtures_dir: Path):
        self.stub = _load_stub()
        stub_provider = self.stub
        self.server = stub_provider.serve(port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.fixtures_dir = fixtures_dir

    def start_server(self, state_dir: Path) -> None:
        _prepare_environment(self.base_url, state_dir, self.fixtures_dir)
        logging.disable(logging.CRITICAL)
        from mcp.types import CallToolRequestParams  # noqa: PLC0415

        import server  # noqa: PLC0415  (environment must be set first)

        server.configure_providers()
        self._server = server
        self._params = CallToolRequestParams

    def fixture(self, name: str) -> str:
        return str(self.fixtures_dir / name)

    def _reset(self, model: str) -> None:
        self.stub.StubHandler.model = model
        self.stub.StubHandler.requests = []
        self.stub.StubHandler.previous_by_thread = {}

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _text(await self._server.handle_call_tool(None, self._params(name=tool, arguments=arguments)))

    async def chat_thread(self, prompts: list[str], files: dict[int, list[str]], model: str = "ci-local") -> list[dict]:
        """Run a chat thread, attaching ``files[turn]`` (1-based) on that turn."""
        self._reset("stub-small" if model == "ci-small" else "stub-model")
        continuation_id = None
        for turn, prompt in enumerate(prompts, start=1):
            arguments: dict[str, Any] = {
                "prompt": prompt,
                "model": model,
                "working_directory_absolute_path": str(self.fixtures_dir),
            }
            if files.get(turn):
                arguments["absolute_file_paths"] = files[turn]
            if continuation_id:
                arguments["continuation_id"] = continuation_id
            response = await self.call("chat", arguments)
            continuation_id = continuation_id or (response.get("continuation_offer") or {}).get("continuation_id")
            if not continuation_id:
                raise RuntimeError(f"chat turn {turn} returned no continuation_id: {response}")
        return list(self.stub.StubHandler.requests)

    async def consensus_pair(self, proposal: str, files: list[str]) -> list[dict]:
        self._reset("stub-model")
        step = {
            "step": proposal,
            "total_steps": 2,
            "findings": "Two stances on one proposal",
            "relevant_files": files,
            "models": [{"model": "ci-local", "stance": "for"}, {"model": "ci-local", "stance": "against"}],
        }
        first = await self.call("consensus", {**step, "step_number": 1, "next_step_required": True})
        thread_id = first["continuation_offer"]["continuation_id"]
        await self.call(
            "consensus",
            {
                **step,
                "step": "first stance heard",
                "step_number": 2,
                "next_step_required": False,
                "continuation_id": thread_id,
            },
        )
        return list(self.stub.StubHandler.requests)

    async def expert_requests(self) -> list[dict]:
        """S7: one single-step expert request each for codereview, debug and thinkdeep."""
        self._reset("stub-model")
        files = [self.fixture("alpha.py"), self.fixture("beta.py")]
        common = {
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "files_checked": files,
            "relevant_files": files,
            "relevant_context": ["Ledger.post", "Ledger.reverse", "settle", "reconcile", "audit_trail"],
            "model": "ci-local",
        }
        await self.call(
            "codereview",
            {
                **common,
                "step": "Review the ledger",
                "findings": "reverse() skips audit",
                "review_validation_type": "external",
            },
        )
        await self.call(
            "debug",
            {
                **common,
                "step": "Why do reversals drift",
                "findings": "rounding in settle()",
                "hypothesis": "settle rounds per line",
                "confidence": "medium",
            },
        )
        await self.call(
            "thinkdeep",
            {
                **common,
                "step": "Should the ledger be append-only",
                "findings": "reversals rewrite rows",
                "confidence": "medium",
            },
        )
        return list(self.stub.StubHandler.requests)


def write_fixtures(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, sentinel in FIXTURES.items():
        body = "\n".join(f"def function_{i}(value):\n    return value * {i}  # {name}\n" for i in range(40))
        (directory / name).write_text(f"# {sentinel}\n{body}# {FIXTURE_ENDS[name]}\n", encoding="utf-8")


# -- gates ---------------------------------------------------------------------------


def _pairs(texts: list[str]) -> list[dict[str, Any]]:
    pairs = []
    for j, (previous, current) in enumerate(zip(texts, texts[1:], strict=False), start=1):
        pairs.append({"j": j, "lcp": lcp(previous, current), "turn_block_end": end_of_last_turn_block(previous)})
    return pairs


def gate_completeness(
    texts: list[str], prompts: list[str], sentinels_by_turn: dict[int, list[str]], reply: str
) -> list[str]:
    """G0: violations, each naming the request and what it carried the wrong number of times."""
    problems = []
    for index, text in enumerate(texts, start=1):
        for k in range(1, index + 1):
            count = text.count(prompts[k - 1])
            if count != 1:
                problems.append(f"r{index}: user prompt {k} appears {count}x")
        if text.count(reply) != index - 1:
            problems.append(f"r{index}: stub reply appears {text.count(reply)}x, expected {index - 1}")
        attached = [s for turn, sentinels in sentinels_by_turn.items() if turn <= index for s in sentinels]
        for sentinel in dict.fromkeys(attached):
            count = text.count(sentinel)
            if count != 1:
                problems.append(f"r{index}: file {sentinel} appears {count}x")
    return problems


def gate_reaches_past_last_turn(pairs: list[dict[str, Any]]) -> list[str]:
    return [
        f"pair {p['j']}: L={p['lcp']} < last turn block end {p['turn_block_end']}"
        for p in pairs
        if p["turn_block_end"] is not None and p["lcp"] < p["turn_block_end"]
    ]


def gate_append_only(texts: list[str]) -> list[str]:
    return [
        f"pair {j}: r{j + 1} does not start with r{j} (L={lcp(a, b)} of {len(a)})"
        for j, (a, b) in enumerate(zip(texts, texts[1:], strict=False), start=1)
        if not b.startswith(a)
    ]


def gate_consults_share_context(consults: list[str], proposal: str, fixture: str) -> list[str]:
    """G5: the two consults share everything through the proposal and the end of its context file."""
    if len(consults) != 2:
        return [f"expected 2 consult requests, recorded {len(consults)}"]
    first = consults[0]
    end_marker = FIXTURE_ENDS[fixture]
    if proposal not in first or end_marker not in first:
        return ["the first consult lacks the proposal or the end of its context file"]
    must_share = max(first.find(proposal) + len(proposal), first.find(end_marker) + len(end_marker))
    shared = lcp(first, consults[1])
    if shared < must_share:
        return [f"consults first differ at {shared}, before the end of proposal/context at {must_share}"]
    return []


def _result(problems: list[str]) -> dict[str, Any]:
    return {"passed": not problems, "problems": problems[:10], "problem_count": len(problems)}


# -- run -----------------------------------------------------------------------------


def _chat_prompts(tag: str, count: int, filler: int = 0) -> list[str]:
    pad = " ".join(f"word{i}" for i in range(filler))
    return [f"{tag} turn {k}: please comment briefly. {pad}".rstrip() for k in range(1, count + 1)]


async def run_scenarios(harness: Harness, only_s7: bool) -> dict[str, Any]:
    reply = harness.stub.REPLY
    out: dict[str, Any] = {"scenarios": {}, "gates": {}}

    if not only_s7:
        a, b, c = harness.fixture("alpha.py"), harness.fixture("beta.py"), harness.fixture("gamma.py")
        chat_cases = {
            "S1": (_chat_prompts("S1", 4), {}),
            "S2": (_chat_prompts("S2", 4), {k: [a] for k in range(1, 5)}),
            "S3": (_chat_prompts("S3", 5), {1: [a], 3: [b], 5: [c]}),
        }
        sentinel_of = {a: FIXTURES["alpha.py"], b: FIXTURES["beta.py"], c: FIXTURES["gamma.py"]}
        texts_by_case: dict[str, list[str]] = {}
        completeness: list[str] = []
        for name, (prompts, files) in chat_cases.items():
            texts = [ser(r) for r in await harness.chat_thread(prompts, files)]
            texts_by_case[name] = texts
            sentinels = {turn: [sentinel_of[f] for f in paths] for turn, paths in files.items()}
            completeness += [f"{name} {p}" for p in gate_completeness(texts, prompts, sentinels, reply)]
            out["scenarios"][name] = {"requests": len(texts), "pairs": _pairs(texts), "schr": schr(texts)}

        s5_texts = [
            ser(r) for r in await harness.chat_thread(_chat_prompts("S5", 12, filler=250), {}, model="ci-small")
        ]
        s5_pairs = _pairs(s5_texts)
        out["scenarios"]["S5"] = {
            "requests": len(s5_texts),
            "pairs": s5_pairs,
            "schr": schr(s5_texts),
            # S5 exists to exercise the history budget; without an overflow G7 measures nothing.
            "history_overflowed": any(HISTORY_TRUNCATED in t for t in s5_texts),
        }

        proposal = "S6 proposal: move the ledger to an append-only event log."
        consults = [ser(r) for r in await harness.consensus_pair(proposal, [a])]
        shared = lcp(consults[0], consults[1]) if len(consults) == 2 else 0
        end_marker = FIXTURE_ENDS["alpha.py"]
        context_end = consults[0].find(end_marker) + len(end_marker) if consults else -1
        proposal_end = consults[0].find(proposal) + len(proposal) if consults else -1
        must_share = max(context_end, proposal_end)
        out["scenarios"]["S6"] = {"requests": len(consults), "lcp": shared, "must_share_through": must_share}

        out["gates"]["G0"] = _result(completeness)
        out["gates"]["G1"] = _result(
            [f"S1 {p}" for p in gate_reaches_past_last_turn(out["scenarios"]["S1"]["pairs"])]
            + [f"S2 {p}" for p in gate_reaches_past_last_turn(out["scenarios"]["S2"]["pairs"])]
        )
        out["gates"]["G2"] = _result([f"S3 {p}" for p in gate_reaches_past_last_turn(out["scenarios"]["S3"]["pairs"])])
        out["gates"]["G3"] = _result([f"{n} {p}" for n, t in texts_by_case.items() for p in gate_append_only(t)])
        out["gates"]["G5"] = _result(gate_consults_share_context(consults, proposal, "alpha.py"))
        broken = [p for p in s5_pairs if p["turn_block_end"] is not None and p["lcp"] < p["turn_block_end"]]
        g7 = []
        if not out["scenarios"]["S5"]["history_overflowed"]:
            g7.append("S5 never overflowed the history budget, so it cannot test compaction")
        if len(broken) > G7_MAX_BROKEN_PAIRS:
            g7.append(
                f"{len(broken)} of {len(s5_pairs)} pairs break before the last turn block (max {G7_MAX_BROKEN_PAIRS})"
            )
        out["gates"]["G7"] = _result(g7)

    s7 = await harness.expert_requests()
    out["scenarios"]["S7"] = {
        "requests": len(s7),
        "sha256": hashlib.sha256(json.dumps(s7, sort_keys=True).encode()).hexdigest(),
    }
    return out


def _s7_hash(fixtures_dir: Path, seed: str) -> str:
    """Run S7 in a fresh interpreter under PYTHONHASHSEED=seed and return its request hash."""
    env = {**os.environ, "PYTHONHASHSEED": seed}
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--only-s7", "--fixtures", str(fixtures_dir), "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(completed.stdout)["scenarios"]["S7"]["sha256"]


def gate_hash_seed(fixtures_dir: Path) -> dict[str, Any]:
    """G4, with a same-seed control first."""
    control = [_s7_hash(fixtures_dir, HASH_SEEDS[0]) for _ in range(2)]
    if control[0] != control[1]:
        return {
            "passed": False,
            "harness_error": "same-seed control runs disagree; S7 is nondeterministic for another reason",
            "problems": [],
            "problem_count": 0,
        }
    hashes = {seed: _s7_hash(fixtures_dir, seed) for seed in HASH_SEEDS[1:]}
    hashes[HASH_SEEDS[0]] = control[0]
    problems = [
        f"seed {seed}: {h[:12]} != seed 1: {control[0][:12]}" for seed, h in sorted(hashes.items()) if h != control[0]
    ]
    return {**_result(problems), "hashes": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--fixtures", type=Path, help="fixture directory to reuse (created when absent)")
    parser.add_argument("--only-s7", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="pal-cache-prefix-") as tmp:
        fixtures_dir = args.fixtures or Path(tmp) / "fixtures"
        if not fixtures_dir.exists():
            write_fixtures(fixtures_dir)
        harness = Harness(fixtures_dir)
        try:
            harness.start_server(Path(tmp) / "state")
            report = asyncio.run(run_scenarios(harness, args.only_s7))
        finally:
            harness.server.shutdown()
        if not args.only_s7:
            report["gates"]["G4"] = gate_hash_seed(fixtures_dir)
            report["gates"] = dict(sorted(report["gates"].items()))

    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for name, scenario in report["scenarios"].items():
        summary = {k: v for k, v in scenario.items() if k != "pairs"}
        lcps = [p["lcp"] for p in scenario.get("pairs", [])]
        print(f"{name}: {summary}" + (f" L_j={lcps}" if lcps else ""))
    for name, gate in report["gates"].items():
        state = "PASS" if gate["passed"] else ("HARNESS ERROR" if gate.get("harness_error") else "FAIL")
        print(f"{name}: {state} {gate.get('harness_error') or ''}")
        for problem in gate.get("problems", [])[:3]:
            print(f"    {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
