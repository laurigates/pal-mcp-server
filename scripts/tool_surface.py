#!/usr/bin/env python3
"""Measure and pin PAL's MCP tool surface.

The surface a client pays for has two channels. The *eager* channel is what a
deferring client (Claude Code, for one) loads at handshake: tool names plus
descriptions. The *full* channel adds every ``inputSchema``, which such a
client fetches per tool at point of use, and which a non-deferring client
loads up front.

Subcommands
-----------
measure   Per-tool character counts and channel totals. ``--disabled`` applies
          a DISABLED_TOOLS preset so the cost of the shipped default can be
          quoted; ``--json`` emits the same numbers machine-readably.
snapshot  Write the structural surface (names, parameter names, types, enums,
          defaults, required lists; no prose) to a JSON file.
check     Diff the live structural surface against a snapshot; exit 1 on any
          difference. This is the gate that lets description text change
          freely while proving nothing a client keys on has moved.

Run from the repo root: ``uv run python scripts/tool_surface.py measure``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# Default preset from .env.example / README.md ("Enabling Additional Tools").
SHIPPED_DEFAULT_DISABLED = "analyze,refactor,testgen,secaudit,docgen,tracer"


def _load_tools() -> dict[str, Any]:
    """Import the registry the way the server builds it, with DISABLED_TOOLS cleared."""
    os.environ["DISABLED_TOOLS"] = ""
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    sys.path.insert(0, os.getcwd())
    import logging

    logging.disable(logging.CRITICAL)
    from server import TOOLS  # noqa: PLC0415  (import after env setup on purpose)

    return dict(TOOLS)


def _structure(schema: dict[str, Any]) -> dict[str, Any]:
    """Everything in a schema except free prose."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "description"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(schema)


def _rows(tools: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(tools):
        tool = tools[name]
        description = tool.get_description()
        schema = tool.get_input_schema()
        rows.append(
            {
                "tool": name,
                "name_chars": len(name),
                "description_chars": len(description),
                "schema_chars": len(json.dumps(schema, ensure_ascii=False)),
            }
        )
    return rows


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    names = sum(r["name_chars"] for r in rows)
    descriptions = sum(r["description_chars"] for r in rows)
    schemas = sum(r["schema_chars"] for r in rows)
    return {
        "tools": len(rows),
        "name_chars": names,
        "description_chars": descriptions,
        "schema_chars": schemas,
        "eager_chars": names + descriptions,
        "full_chars": names + descriptions + schemas,
    }


def cmd_measure(args: argparse.Namespace) -> int:
    tools = _load_tools()
    rows = _rows(tools)
    disabled = {t.strip() for t in (args.disabled or "").split(",") if t.strip()}
    unknown = disabled - set(tools)
    if unknown:
        print(f"unknown tools in --disabled: {sorted(unknown)}", file=sys.stderr)
        return 2
    enabled_rows = [r for r in rows if r["tool"] not in disabled]

    all_totals = _totals(rows)
    enabled_totals = _totals(enabled_rows)

    if args.json:
        print(
            json.dumps(
                {"per_tool": rows, "all": all_totals, "enabled": enabled_totals, "disabled": sorted(disabled)},
                indent=2,
            )
        )
        return 0

    print(f"{'tool':<12} {'desc':>6} {'schema':>7}   state")
    for r in rows:
        state = "disabled" if r["tool"] in disabled else ""
        print(f"{r['tool']:<12} {r['description_chars']:>6} {r['schema_chars']:>7}   {state}")
    print()
    for label, t in (("all tools", all_totals), ("enabled", enabled_totals)):
        print(
            f"{label:<10} n={t['tools']:<3} eager={t['eager_chars']:>6}  "
            f"(names {t['name_chars']} + descriptions {t['description_chars']})  full={t['full_chars']:>6}"
        )
    if disabled:
        e_cut = 1 - enabled_totals["eager_chars"] / all_totals["eager_chars"]
        f_cut = 1 - enabled_totals["full_chars"] / all_totals["full_chars"]
        print(f"preset cut  eager -{e_cut:.1%}  full -{f_cut:.1%}  (disabled: {','.join(sorted(disabled))})")
    return 0


def _snapshot(tools: dict[str, Any]) -> dict[str, Any]:
    return {name: _structure(tools[name].get_input_schema()) for name in sorted(tools)}


def cmd_snapshot(args: argparse.Namespace) -> int:
    snap = _snapshot(_load_tools())
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {args.out} ({len(snap)} tools)")
    return 0


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"+ {path}.{key}")
            elif key not in b:
                out.append(f"- {path}.{key}")
            else:
                _diff(a[key], b[key], f"{path}.{key}", out)
    elif a != b:
        out.append(f"~ {path}: {json.dumps(a, ensure_ascii=False)} -> {json.dumps(b, ensure_ascii=False)}")


def cmd_check(args: argparse.Namespace) -> int:
    with open(args.against, encoding="utf-8") as fh:
        expected = json.load(fh)
    live = _snapshot(_load_tools())
    findings: list[str] = []
    _diff(expected, live, "", findings)
    if findings:
        print("structural surface differs from snapshot:")
        for line in findings:
            print("  " + line.replace("..", ".", 1).lstrip("."))
        return 1
    print(f"structural surface matches snapshot ({len(live)} tools)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    measure = sub.add_parser("measure", help="character counts per tool and per channel")
    measure.add_argument("--disabled", metavar="A,B,C", help="apply a DISABLED_TOOLS preset")
    measure.add_argument("--preset", action="store_true", help=f"shorthand for --disabled {SHIPPED_DEFAULT_DISABLED}")
    measure.add_argument("--json", action="store_true")

    snapshot = sub.add_parser("snapshot", help="write the structural surface to a JSON file")
    snapshot.add_argument("--out", required=True)

    check = sub.add_parser("check", help="diff the live structural surface against a snapshot")
    check.add_argument("--against", required=True)

    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["measure"])
    if args.command == "measure":
        if args.preset:
            args.disabled = SHIPPED_DEFAULT_DISABLED
        return cmd_measure(args)
    if args.command == "snapshot":
        return cmd_snapshot(args)
    if args.command == "check":
        return cmd_check(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
