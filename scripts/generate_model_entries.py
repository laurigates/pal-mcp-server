#!/usr/bin/env python3
"""Append catalog-derived entries for untriaged models to conf/*_models.json.

The audit (scripts/audit_model_registry.py) reports live catalog models that a
config does not have as MISSING candidates. This script writes one entry per
candidate, taking every field from the catalog and none from judgment:

  * ``context_window`` / ``max_output_tokens`` -- only when the catalog states
    a positive number; an absent limit is left absent, never guessed.
  * ``supports_images``, ``supports_function_calling``,
    ``supports_extended_thinking``, ``supports_temperature`` -- per the field
    mapping in .claude/skills/model-registry-audit/SKILL.md.
  * ``enabled_by_default: false`` -- so a generated entry never changes auto
    mode, the ranked summary or alias resolution until a human promotes it.

It never writes ``intelligence_score`` or ``aliases``; those stay human
decisions made when an entry is promoted.

Entries are spliced into the file text before the closing bracket of the
``models`` array rather than re-serialising the whole file, so hand-formatted
entries elsewhere keep their layout. The result is written for review in a
diff; nothing is committed.

Exit codes: 0 success, 2 a catalog or config could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_model_registry import (  # noqa: E402 - needs the sys.path entry above
    CONF_DIR,
    TARGETS,
    Target,
    candidate_facts,
    claimed_names,
    live_index,
    load_catalogs,
    read_config,
)


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def entry_from_facts(facts: dict[str, Any], source: str) -> dict[str, Any]:
    """Build a disabled registry entry from one catalog record."""
    inputs = [m.lower() for m in facts.get("input_modalities") or []]
    if source == "openrouter":
        params = set(facts.get("supported_parameters") or [])
        images = "image" in inputs
        tools = "tools" in params
        reasoning = "reasoning" in params or "include_reasoning" in params
        # OpenRouter lists the sampling parameters a model accepts; one that
        # omits temperature rejects it.
        temperature: bool | None = "temperature" in params if params else None
    else:
        images = bool(facts.get("attachment")) or "image" in inputs
        tools = bool(facts.get("tool_call"))
        reasoning = bool(facts.get("reasoning"))
        temperature = facts.get("temperature") if isinstance(facts.get("temperature"), bool) else None

    entry: dict[str, Any] = {"model_name": facts["id"]}
    if facts.get("name"):
        entry["description"] = facts["name"]
    for cfg_field in ("context_window", "max_output_tokens"):
        value = _positive_int(facts.get(cfg_field))
        if value is not None:
            entry[cfg_field] = value
    entry["supports_images"] = images
    entry["supports_function_calling"] = tools
    entry["supports_extended_thinking"] = reasoning
    if temperature is False:
        entry["supports_temperature"] = False
        entry["temperature_constraint"] = "fixed"
    entry["enabled_by_default"] = False
    return entry


def splice_entries(text: str, entries: list[dict[str, Any]]) -> str:
    """Insert ``entries`` at the end of the top-level ``models`` array in ``text``.

    Relies on ``models`` being the last key of the document, which holds for
    every conf/*_models.json; anything else raises rather than guessing.
    """
    if not entries:
        return text
    original = json.loads(text)
    body = text.rstrip()
    if not body.endswith("}"):
        raise ValueError("config does not end with a closing brace")
    body = body[:-1].rstrip()
    if not body.endswith("]"):
        raise ValueError("'models' is not the last key of the config")
    head = body[:-1].rstrip()

    rendered = ",\n".join(
        "\n".join("    " + line for line in json.dumps(entry, indent=2, ensure_ascii=False).splitlines())
        for entry in entries
    )
    separator = "\n" if head.endswith("[") else ",\n"
    result = f"{head}{separator}{rendered}\n  ]\n}}\n"

    # The splice is text surgery on JSON; prove it produced exactly the
    # original document plus the new entries before anyone writes it.
    expected = dict(original)
    expected["models"] = list(original["models"]) + entries
    if json.loads(result) != expected:
        raise ValueError("splice changed the document beyond appending entries")
    return result


def generate_for(target: Target, catalogs: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    path = CONF_DIR / target.filename
    models, err = read_config(path)
    if err:
        return [], err
    live, _ = live_index(target, catalogs)
    if not live:
        return [], f"catalog slice for {target.provider_id or target.source} came back empty"
    candidates = sorted(candidate_facts(live, claimed_names(models)), key=lambda f: f["id"])
    return [entry_from_facts(facts, target.source) for facts in candidates], None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, help="read/write fetched catalogs here")
    ap.add_argument("--offline", action="store_true", help="use only cached catalogs (requires --cache-dir)")
    ap.add_argument("--only", help="generate for a single config file, e.g. openrouter_models.json")
    ap.add_argument("--dry-run", action="store_true", help="print the entries instead of writing the configs")
    args = ap.parse_args()

    if args.offline and not args.cache_dir:
        print("--offline requires --cache-dir", file=sys.stderr)
        return 2

    targets = [t for t in TARGETS if t.source != "unverifiable" and (not args.only or t.filename == args.only)]
    if not targets:
        print(f"no catalog-backed config named {args.only}", file=sys.stderr)
        return 2

    catalogs = load_catalogs(args.cache_dir, args.offline)
    status = 0
    for target in targets:
        entries, err = generate_for(target, catalogs)
        if err:
            print(f"{target.filename}: ERROR={err}", file=sys.stderr)
            status = 2
            continue
        print(f"{target.filename}: GENERATED={len(entries)}")
        if not entries:
            continue
        if args.dry_run:
            for entry in entries:
                print(json.dumps(entry, ensure_ascii=False))
            continue
        path = CONF_DIR / target.filename
        path.write_text(splice_entries(path.read_text(), entries))
    return status


if __name__ == "__main__":
    sys.exit(main())
