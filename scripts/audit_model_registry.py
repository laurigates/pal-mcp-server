#!/usr/bin/env python3
"""Audit conf/*_models.json against live provider catalogs.

Mechanical drift detection only. Every finding is derived from a catalog the
script fetched itself; nothing here guesses at a model identifier. Judgment
calls -- whether a new model is worth exposing, what intelligence_score it
deserves, which alias it should inherit -- are left to the caller (see
.claude/skills/model-registry-audit/SKILL.md).

Two public, keyless catalogs back the audit:

  openrouter  https://openrouter.ai/api/v1/models  -- authoritative for the
              OpenRouter config, and the only source carrying an explicit
              ``expiration_date`` deprecation signal.
  models.dev  https://models.dev/api.json          -- community-maintained
              cross-provider catalog covering google, openai, xai and the
              opencode zen gateway.

Because models.dev is community-maintained, absence from it is reported as
``review`` rather than ``confirmed``: it omits live models often enough that
treating a miss as proof of removal would delete working entries.

Exit codes: 0 no drift, 1 drift found (with --fail-on-drift), 2 audit could not run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = REPO_ROOT / "conf"

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
MODELSDEV_URL = "https://models.dev/api.json"

USER_AGENT = "pal-mcp-server-model-audit/1 (+https://github.com/laurigates/pal-mcp-server)"

# Mirrors providers/shared/model_capabilities.py :: ModelCapabilities. Fields
# outside this set are dropped by providers/registries/base.py's
# CAPABILITY_FIELD_NAMES filter, so a typo there is silent at runtime -- which is
# exactly why the audit checks for it.
CAPABILITY_FIELDS = {
    "provider",
    "model_name",
    "friendly_name",
    "intelligence_score",
    "description",
    "aliases",
    "context_window",
    "max_output_tokens",
    "max_thinking_tokens",
    "supports_extended_thinking",
    "supports_system_prompts",
    "supports_streaming",
    "supports_function_calling",
    "supports_images",
    "supports_json_mode",
    "supports_temperature",
    "use_openai_response_api",
    "default_reasoning_effort",
    "allow_code_generation",
    "max_image_size_mb",
    "temperature_constraint",
}


@dataclass(frozen=True)
class Target:
    """One config file and the catalog slice that can verify it."""

    filename: str
    source: str  # "openrouter" | "modelsdev" | "unverifiable"
    provider_id: str = ""  # models.dev provider key
    label: str = ""
    reason: str = ""  # why unverifiable, when applicable
    module: str = ""  # providers/<module>, checked for stale hardcoded ids


TARGETS: tuple[Target, ...] = (
    Target("openrouter_models.json", "openrouter", label="OpenRouter", module="openrouter.py"),
    Target("gemini_models.json", "modelsdev", "google", "Gemini", module="gemini.py"),
    Target("openai_models.json", "modelsdev", "openai", "OpenAI", module="openai.py"),
    Target("xai_models.json", "modelsdev", "xai", "xAI", module="xai.py"),
    # "opencode-go", not "opencode": models.dev carries both, and they are
    # different gateways. The bare "opencode" slice (93 models) lacks the whole
    # mimo/qwen3.7 family this config serves, so pointing at it reported six
    # live models as withdrawn.
    Target(
        "opencode_go_models.json",
        "modelsdev",
        "opencode-go",
        "OpenCode Zen (go)",
        module="opencode_go.py",
    ),
    Target(
        "dial_models.json",
        "unverifiable",
        label="DIAL",
        reason="enterprise aggregator, no public keyless catalog",
        module="dial.py",
    ),
    Target(
        "custom_models.json",
        "unverifiable",
        label="Custom",
        reason="user-supplied local endpoints (Ollama, vLLM)",
    ),
    Target(
        "azure_models.json",
        "unverifiable",
        label="Azure",
        reason="per-deployment names chosen by the operator",
    ),
)

# Non-chat model families PAL never routes to. Matched as substrings against the
# catalog id, and used only to filter *candidate additions* -- never to justify
# removing something already configured.
NON_CHAT_MARKERS = (
    "embedding",
    "embed-",
    "-tts",
    "tts-",
    "whisper",
    "-stt",
    "transcribe",
    "moderation",
    "rerank",
    "-image",
    "image-",
    "imagen",
    "veo-",
    "-video",
    "guard",
    "-live",
    "realtime",
    "computer-use",
    "-search-preview",
)


# A model id as it appears in source: lowercase, at least one separator, and
# (checked separately) at least one digit. Deliberately narrow -- it runs over
# every string literal in a provider module, so a loose pattern would flag URLs
# and header names.
MODEL_ID_SHAPE = re.compile(r"^[a-z][a-z0-9]*(?:[./-][a-z0-9.]+)+$")

# Literals that pass MODEL_ID_SHAPE but are plainly not model ids. Encodings and
# protocol versions are the whole population: "utf-8" is lowercase, separated,
# and carries a digit, so the shape filter alone reports it. Extend this rather
# than loosening the pattern -- a shape wide enough to exclude "utf-8" also
# excludes "gpt-5".
NON_MODEL_LITERALS = frozenset(
    {
        "utf-8",
        "utf-16",
        "utf-32",
        "latin-1",
        "iso-8859-1",
        "sha-1",
        "sha-256",
        "sha-512",
        "http/1.1",
        "http/2",
        "base-64",
    }
)

# Findings that represent something broken now. Candidate additions are excluded
# on purpose: a nonzero MISSING is the normal state of a curated registry, so
# gating CI on it would make the gate meaningless.
ACTIONABLE_KINDS = ("deprecated", "stale", "alias_collision", "orphan_ref", "schema")


@dataclass
class Finding:
    kind: str  # deprecated | missing | stale | alias_collision | schema | orphan_ref
    config_file: str
    model_name: str
    detail: str
    confidence: str = "confirmed"  # confirmed | review
    extra: dict[str, Any] = field(default_factory=dict)


def fetch_json(url: str, timeout: int = 45) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https literals
        return json.loads(resp.read().decode("utf-8"))


def load_catalogs(cache_dir: Path | None, offline: bool) -> dict[str, Any]:
    """Return both catalogs, reading from and writing through a cache dir."""
    catalogs: dict[str, Any] = {}
    for name, url in (("openrouter", OPENROUTER_URL), ("modelsdev", MODELSDEV_URL)):
        cached = (cache_dir / f"{name}.json") if cache_dir else None
        if offline:
            if not cached or not cached.exists():
                raise SystemExit(f"--offline needs a cached catalog at {cached}")
            catalogs[name] = json.loads(cached.read_text())
            continue
        try:
            catalogs[name] = fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if cached and cached.exists():
                print(f"WARN fetch failed for {name} ({exc}); falling back to cache", file=sys.stderr)
                catalogs[name] = json.loads(cached.read_text())
                continue
            raise SystemExit(f"could not fetch {name} catalog and no cache available: {exc}") from exc
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(catalogs[name]))
    return catalogs


def openrouter_index(catalog: Any) -> dict[str, dict[str, Any]]:
    """Normalise the OpenRouter payload into id -> facts.

    Both ``id`` and ``canonical_slug`` are indexed, and ``:batch`` / ``:free``
    variants are folded onto their base id: a re-slugged model is a rename, not
    a removal, and keying on ``id`` alone would report it as gone.
    """
    out: dict[str, dict[str, Any]] = {}
    for m in catalog.get("data", []):
        top = m.get("top_provider") or {}
        arch = m.get("architecture") or {}
        facts = {
            "id": m.get("id"),
            "name": m.get("name"),
            "context_window": m.get("context_length") or top.get("context_length"),
            "max_output_tokens": top.get("max_completion_tokens"),
            "expiration_date": m.get("expiration_date"),
            "created": m.get("created"),
            "input_modalities": arch.get("input_modalities") or [],
            "output_modalities": arch.get("output_modalities") or [],
            "supported_parameters": m.get("supported_parameters") or [],
            "description": m.get("description") or "",
        }
        for key in filter(None, {m.get("id"), m.get("canonical_slug")}):
            out.setdefault(key, facts)
            out.setdefault(key.split(":", 1)[0], facts)
    return out


def modelsdev_index(catalog: Any, provider_id: str) -> dict[str, dict[str, Any]]:
    provider = catalog.get(provider_id) or {}
    out: dict[str, dict[str, Any]] = {}
    for mid, m in (provider.get("models") or {}).items():
        limit = m.get("limit") or {}
        modalities = m.get("modalities") or {}
        facts = {
            "id": mid,
            "name": m.get("name"),
            "context_window": limit.get("context"),
            "max_output_tokens": limit.get("output"),
            "expiration_date": None,
            "created": m.get("release_date"),
            "last_updated": m.get("last_updated"),
            "input_modalities": modalities.get("input") or [],
            "output_modalities": modalities.get("output") or [],
            "reasoning": m.get("reasoning"),
            "tool_call": m.get("tool_call"),
            "attachment": m.get("attachment"),
            "cost": m.get("cost") or {},
            "description": m.get("description") or "",
        }
        out[mid] = facts
        # models.dev normalises dots to dashes in some ids (claude-opus-4-5);
        # index a dotted variant too so an exact-match lookup does not miss.
        out.setdefault(mid.replace("-", "."), facts)
    return out


def is_chat_model(facts: dict[str, Any]) -> bool:
    mid = (facts.get("id") or "").lower()
    if any(marker in mid for marker in NON_CHAT_MARKERS):
        return False
    outs = [o.lower() for o in facts.get("output_modalities") or []]
    return "text" in outs if outs else True


def is_candidate_id(model_id: str) -> bool:
    """Reject ids that are variants or routing aliases of a real model.

    OpenRouter exposes ``:batch`` / ``:free`` / ``:thinking`` suffixes and
    ``~vendor/x-latest`` wildcard routers alongside the base models. Proposing
    those as additions would bury the genuinely new releases.
    """
    return not model_id.startswith("~") and ":" not in model_id


def read_config(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return [], f"missing config file {path.name}"
    except json.JSONDecodeError as exc:
        return [], f"{path.name} is not valid JSON: {exc}"
    models = data.get("models")
    if not isinstance(models, list):
        return [], f"{path.name} has no 'models' array"
    return models, None


def check_aliases(filename: str, models: list[dict[str, Any]]) -> list[Finding]:
    """An alias pointing at two models in one file resolves unpredictably."""
    owner: dict[str, str] = {}
    out: list[Finding] = []
    for m in models:
        name = m.get("model_name", "?")
        for alias in m.get("aliases", []) or []:
            key = alias.lower()
            if key in owner and owner[key] != name:
                out.append(
                    Finding(
                        "alias_collision",
                        filename,
                        name,
                        f"alias '{alias}' already maps to {owner[key]} in this file",
                    )
                )
            else:
                owner[key] = name
    return out


def check_schema(filename: str, models: list[dict[str, Any]]) -> list[Finding]:
    """Unknown keys are dropped silently by the registry -- surface them."""
    out: list[Finding] = []
    for m in models:
        name = m.get("model_name", "?")
        if not m.get("model_name"):
            out.append(Finding("schema", filename, "?", "entry has no model_name"))
        unknown = sorted(set(m) - CAPABILITY_FIELDS)
        if unknown:
            out.append(
                Finding(
                    "schema",
                    filename,
                    name,
                    f"unknown field(s) {', '.join(unknown)} -- dropped by CAPABILITY_FIELD_NAMES",
                )
            )
    return out


def check_provider_references(filename: str, module: str, models: list[dict[str, Any]]) -> list[Finding]:
    """Flag model ids hardcoded in a provider module that the config no longer has.

    Providers pin canonical ids outside the registry -- ``PRIMARY_MODEL`` /
    ``FALLBACK_MODEL`` and the per-category preference lists. Those are plain
    strings, so when a model leaves the config they keep pointing at it and
    category routing silently resolves to nothing. Nothing else in the suite
    notices: the tests exercise the resolver, not the constants.

    Observed 2026-08-28: ``providers/xai.py`` named ``grok-4-1-fast-reasoning``
    and ``grok-4`` as PRIMARY/FALLBACK after xAI retired both, and
    ``providers/openai.py`` listed the shut-down ``gpt-5-codex`` in three
    preference lists.

    String literals are read from the AST and filtered to model-id shape
    (lowercase, a separator, at least one digit), which across all six provider
    modules yields no false positives.
    """
    path = REPO_ROOT / "providers" / module
    try:
        tree = ast.parse(path.read_text())
    except (FileNotFoundError, SyntaxError):
        return []

    known = {m["model_name"] for m in models if m.get("model_name")}
    known |= {a.lower() for m in models for a in (m.get("aliases") or [])}

    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    out: list[Finding] = []
    for literal in sorted(literals):
        if literal in NON_MODEL_LITERALS:
            continue
        if not MODEL_ID_SHAPE.match(literal) or not any(c.isdigit() for c in literal):
            continue
        if literal in known or literal.lower() in known:
            continue
        out.append(
            Finding(
                "orphan_ref",
                filename,
                literal,
                f"providers/{module} hardcodes it, but it is not in this config",
            )
        )
    return out


def audit_target(target: Target, catalogs: dict[str, Any], top_n: int) -> tuple[list[Finding], dict[str, Any]]:
    path = CONF_DIR / target.filename
    models, err = read_config(path)
    meta: dict[str, Any] = {
        "config_file": target.filename,
        "label": target.label,
        "source": target.source,
        "configured": len(models),
        "error": err,
        "reason": target.reason,
    }
    findings: list[Finding] = []
    if err:
        return findings, meta

    findings.extend(check_aliases(target.filename, models))
    findings.extend(check_schema(target.filename, models))
    if target.module:
        findings.extend(check_provider_references(target.filename, target.module, models))

    if target.source == "unverifiable":
        meta["catalog"] = 0
        return findings, meta

    if target.source == "openrouter":
        live = openrouter_index(catalogs["openrouter"])
        confidence = "confirmed"  # the provider's own live endpoint
    else:
        live = modelsdev_index(catalogs["modelsdev"], target.provider_id)
        confidence = "review"  # community-maintained; absence is not proof

    meta["catalog"] = len({f["id"] for f in live.values() if f.get("id")})
    if not live:
        meta["error"] = f"catalog slice for {target.provider_id or target.source} came back empty"
        return findings, meta

    configured_ids = set()
    for m in models:
        name = m.get("model_name")
        if not name:
            continue
        configured_ids.add(name)
        facts = live.get(name) or live.get(name.split(":", 1)[0])
        if facts is None:
            findings.append(
                Finding(
                    "deprecated",
                    target.filename,
                    name,
                    f"absent from the {target.source} catalog",
                    confidence,
                    {"aliases": m.get("aliases", [])},
                )
            )
            continue
        if facts.get("expiration_date"):
            findings.append(
                Finding(
                    "deprecated",
                    target.filename,
                    name,
                    f"catalog sets expiration_date={facts['expiration_date']}",
                    "confirmed",
                    {"aliases": m.get("aliases", []), "expires": facts["expiration_date"]},
                )
            )
        for cfg_field in ("context_window", "max_output_tokens"):
            have, want = m.get(cfg_field), facts.get(cfg_field)
            if isinstance(want, int) and isinstance(have, int) and want > 0 and have != want:
                findings.append(
                    Finding(
                        "stale",
                        target.filename,
                        name,
                        f"{cfg_field}: configured {have:,} vs catalog {want:,}",
                        confidence,
                        {"field": cfg_field, "configured": have, "catalog": want},
                    )
                )

    # Candidate additions: live chat models this config does not expose, newest
    # first. Ranked, never auto-applied -- picking which to add is judgment.
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for facts in live.values():
        mid = facts.get("id")
        if not mid or mid in seen or mid in configured_ids:
            continue
        seen.add(mid)
        if is_chat_model(facts) and is_candidate_id(mid):
            candidates.append(facts)
    candidates.sort(key=lambda f: str(f.get("created") or ""), reverse=True)
    for facts in candidates[:top_n]:
        findings.append(
            Finding(
                "missing",
                target.filename,
                facts["id"],
                f"live in catalog (released/created {facts.get('created')}), not configured",
                "review",
                {
                    "context_window": facts.get("context_window"),
                    "max_output_tokens": facts.get("max_output_tokens"),
                    "released": facts.get("created"),
                },
            )
        )
    meta["candidates_total"] = len(candidates)
    meta["candidates_shown"] = min(top_n, len(candidates))
    return findings, meta


def emit_text(results: list[tuple[Target, list[Finding], dict[str, Any]]]) -> None:
    print("=== COVERAGE ===")
    for target, _, meta in results:
        note = f" ({meta['reason']})" if meta.get("reason") else ""
        err = f" ERROR={meta['error']}" if meta.get("error") else ""
        print(
            f"{meta['label'] or target.filename}: configured={meta['configured']} "
            f"catalog={meta.get('catalog', 'n/a')} source={meta['source']}{note}{err}"
        )

    for kind, heading in (
        ("deprecated", "DEPRECATED"),
        ("stale", "STALE METADATA"),
        ("alias_collision", "ALIAS COLLISIONS"),
        ("orphan_ref", "STALE PROVIDER REFERENCES"),
        ("schema", "SCHEMA"),
        ("missing", "CANDIDATE ADDITIONS"),
    ):
        rows = [f for _, fs, _ in results for f in fs if f.kind == kind]
        print(f"\n=== {heading} ({len(rows)}) ===")
        for f in rows:
            mark = "" if f.confidence == "confirmed" else " [review]"
            print(f"{f.config_file}: {f.model_name} -- {f.detail}{mark}")

    counts: dict[str, int] = dict.fromkeys(
        ("deprecated", "stale", "alias_collision", "orphan_ref", "schema", "missing"), 0
    )
    for _, fs, _ in results:
        for f in fs:
            counts[f.kind] = counts.get(f.kind, 0) + 1
    print("\n=== SUMMARY ===")
    for k, v in counts.items():
        print(f"{k.upper()}={v}")
    actionable = sum(counts[k] for k in ACTIONABLE_KINDS)
    print(f"ACTIONABLE={actionable}")
    print(f"STATUS={'DRIFT' if actionable else 'CLEAN'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of the text report")
    ap.add_argument("--cache-dir", type=Path, help="read/write fetched catalogs here")
    ap.add_argument("--offline", action="store_true", help="use only cached catalogs (requires --cache-dir)")
    ap.add_argument("--top-n", type=int, default=10, help="candidate additions to list per provider (default 10)")
    ap.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit 1 when deprecated/stale/collision/schema findings exist (candidates alone never fail)",
    )
    ap.add_argument("--only", help="audit a single config file, e.g. openrouter_models.json")
    args = ap.parse_args()

    if args.offline and not args.cache_dir:
        print("--offline requires --cache-dir", file=sys.stderr)
        return 2

    targets = [t for t in TARGETS if not args.only or t.filename == args.only]
    if not targets:
        print(f"no such config: {args.only}", file=sys.stderr)
        return 2

    needs_network = any(t.source != "unverifiable" for t in targets)
    catalogs = load_catalogs(args.cache_dir, args.offline) if needs_network else {}

    results = [(t, *audit_target(t, catalogs, args.top_n)) for t in targets]

    if args.json:
        print(
            json.dumps(
                {
                    "coverage": [meta for _, _, meta in results],
                    "findings": [
                        {
                            "kind": f.kind,
                            "config_file": f.config_file,
                            "model_name": f.model_name,
                            "detail": f.detail,
                            "confidence": f.confidence,
                            **f.extra,
                        }
                        for _, fs, _ in results
                        for f in fs
                    ],
                },
                indent=2,
            )
        )
    else:
        emit_text(results)

    if any(meta.get("error") for _, _, meta in results):
        return 2
    actionable = sum(1 for _, fs, _ in results for f in fs if f.kind in ACTIONABLE_KINDS)
    return 1 if (args.fail_on_drift and actionable) else 0


if __name__ == "__main__":
    sys.exit(main())
