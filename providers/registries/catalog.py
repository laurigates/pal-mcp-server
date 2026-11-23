"""Catalog facts for model registries, fetched from a provider's own catalog (#240).

ADR-0001 splits each registry entry into catalog facts (limits, modalities,
tool calling, reasoning) and curated judgment (aliases, ``intelligence_score``,
``enabled_by_default``, ...). This module holds the provider-independent
pieces of phase 1, the OpenRouter catalog:

* ``validate_openrouter_catalog`` / ``openrouter_index`` -- turn the
  ``GET /api/v1/models`` payload into id -> facts, dropping malformed records.
* ``merge_openrouter_catalog`` -- lay the bundled ``conf/openrouter_models.json``
  entries over those facts. The bundled entry supplies every curated field;
  the catalog supplies the fact fields in ``CATALOG_FACT_FIELDS``. A curated
  entry the catalog no longer lists is disabled, not deleted, and a catalog
  model with no curated entry arrives disabled.
* ``read_cache`` / ``write_cache`` -- the on-disk fallback under the state
  directory, written atomically.
* ``fetch_json`` -- one stdlib request; ``urllib`` honours ``HTTPS_PROXY``.

The fact mapping mirrors ``scripts/generate_model_entries.py :: entry_from_facts``
and ``scripts/audit_model_registry.py :: openrouter_index``. Those scripts run
in CI without the project's dependencies installed, so they cannot import this
module; ``tests/test_registry_catalog.py`` pins the two copies to the same output.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from utils.env import get_env
from utils.state_dir import get_state_dir

logger = logging.getLogger(__name__)

OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"
DYNAMIC_REGISTRY_ENV = "PAL_DYNAMIC_REGISTRY"
CACHE_SUBDIR = "registry-cache"
FETCH_TIMEOUT_SECONDS = 15.0

#: Fields the catalog overrides on a curated entry. ``supports_temperature``
#: stays curated in this phase: the bundled files pair it with a hand-chosen
#: ``temperature_constraint``, and flipping one without the other would send
#: a temperature the model rejects.
CATALOG_FACT_FIELDS = (
    "context_window",
    "max_output_tokens",
    "supports_images",
    "supports_function_calling",
    "supports_extended_thinking",
)

#: Upper bound for any token limit taken from a catalog. The largest window in
#: the bundled registries is about 2M; a value past this is a catalog defect,
#: and the curated value is kept instead.
MAX_TOKEN_LIMIT = 20_000_000

#: Id fragments of models that do not answer chat completions. Kept in step
#: with ``NON_CHAT_MARKERS`` in scripts/audit_model_registry.py.
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

_FALSE_VALUES = {"0", "false", "no", "off"}


def dynamic_registry_enabled() -> bool:
    """On unless ``PAL_DYNAMIC_REGISTRY`` is 0/false/no/off (PRD requirement 7)."""
    value = get_env(DYNAMIC_REGISTRY_ENV)
    return value is None or value.strip().lower() not in _FALSE_VALUES


# ----------------------------------------------------------------------
# Catalog normalisation
# ----------------------------------------------------------------------
def _bounded_limit(value: Any, ceiling: int = MAX_TOKEN_LIMIT) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 < value <= ceiling else None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def validate_openrouter_catalog(payload: Any) -> list[dict[str, Any]]:
    """Return the well-formed model records of an OpenRouter catalog payload.

    Raises ``ValueError`` when the payload itself has the wrong shape, so a
    broken response never replaces a good cache. A single malformed record is
    dropped, not partially applied (PRD requirement 5).
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter catalog payload has no 'data' list")
    records = []
    for record in payload["data"]:
        if not isinstance(record, dict):
            continue
        model_id = record.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        if not isinstance(record.get("top_provider") or {}, dict):
            continue
        if not isinstance(record.get("architecture") or {}, dict):
            continue
        records.append(record)
    if not records:
        raise ValueError("OpenRouter catalog payload lists no valid models")
    return records


def openrouter_facts(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise one OpenRouter record (same keys as the audit script's index)."""
    top = record.get("top_provider") or {}
    arch = record.get("architecture") or {}
    return {
        "id": record.get("id"),
        "name": record.get("name"),
        "context_window": record.get("context_length") or top.get("context_length"),
        "max_output_tokens": top.get("max_completion_tokens"),
        "expiration_date": record.get("expiration_date"),
        "created": record.get("created"),
        "input_modalities": _str_list(arch.get("input_modalities")),
        "output_modalities": _str_list(arch.get("output_modalities")),
        "supported_parameters": _str_list(record.get("supported_parameters")),
        "description": record.get("description") or "",
    }


def openrouter_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map lowercased id, canonical slug and ``:variant``-stripped id to facts.

    A re-slugged model is a rename, not a removal, so ``canonical_slug`` is
    indexed too, and ``:free`` / ``:batch`` variants fold onto their base id.
    """
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        facts = openrouter_facts(record)
        for key in filter(None, {record.get("id"), record.get("canonical_slug")}):
            if not isinstance(key, str):
                continue
            out.setdefault(key.lower(), facts)
            out.setdefault(key.split(":", 1)[0].lower(), facts)
    return out


def fact_fields(facts: dict[str, Any]) -> dict[str, Any]:
    """The ``CATALOG_FACT_FIELDS`` values one catalog record supports.

    A limit the catalog omits, or states outside ``(0, MAX_TOKEN_LIMIT]``, is
    left out so the curated value stands. An output limit above the context
    window is dropped for the same reason.
    """
    params = set(facts.get("supported_parameters") or [])
    inputs = [m.lower() for m in facts.get("input_modalities") or []]
    out: dict[str, Any] = {}
    context = _bounded_limit(facts.get("context_window"))
    if context is not None:
        out["context_window"] = context
    output = _bounded_limit(facts.get("max_output_tokens"), context or MAX_TOKEN_LIMIT)
    if output is not None:
        out["max_output_tokens"] = output
    out["supports_images"] = "image" in inputs
    out["supports_function_calling"] = "tools" in params
    out["supports_extended_thinking"] = "reasoning" in params or "include_reasoning" in params
    return out


def entry_from_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """A disabled registry entry for a catalog model with no curated entry."""
    entry: dict[str, Any] = {"model_name": facts["id"]}
    if isinstance(facts.get("name"), str) and facts["name"]:
        entry["description"] = facts["name"]
    entry.update(fact_fields(facts))
    params = facts.get("supported_parameters") or []
    # OpenRouter lists the sampling parameters a model accepts; one that omits
    # temperature rejects it.
    if params and "temperature" not in params:
        entry["supports_temperature"] = False
        entry["temperature_constraint"] = "fixed"
    entry["enabled_by_default"] = False
    return entry


def _is_new_model_candidate(facts: dict[str, Any]) -> bool:
    model_id = (facts.get("id") or "").lower()
    if model_id.startswith("~") or ":" in model_id:
        return False
    if any(marker in model_id for marker in NON_CHAT_MARKERS):
        return False
    if facts.get("expiration_date"):
        return False
    outputs = [o.lower() for o in facts.get("output_modalities") or []]
    return "text" in outputs if outputs else True


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------
def merge_openrouter_catalog(
    curated: list[Any], records: list[dict[str, Any]]
) -> tuple[list[Any], dict[str, list[str]]]:
    """Return the effective entries and a summary of what the catalog changed.

    ``curated`` is the bundled ``models`` list; it is not mutated. Entries
    routed to another provider (a ``provider`` field other than openrouter)
    pass through untouched, because the OpenRouter catalog says nothing about
    them. The summary has ``withdrawn`` (curated entries the catalog dropped
    that were enabled until now), ``absent`` (every curated OpenRouter entry
    the catalog does not list, enabled or not) and ``added`` (catalog models
    appended disabled).
    """
    index = openrouter_index(records)
    effective: list[Any] = []
    withdrawn: list[str] = []
    absent: list[str] = []
    claimed: set[str] = set()

    for raw in curated:
        if not isinstance(raw, dict) or not isinstance(raw.get("model_name"), str):
            effective.append(raw)
            continue
        entry = dict(raw)
        name = entry["model_name"]
        claimed.add(name.lower())
        aliases = entry.get("aliases")
        if isinstance(aliases, list):
            claimed.update(a.lower() for a in aliases if isinstance(a, str))
        provider = entry.get("provider")
        if isinstance(provider, str) and provider.lower() != "openrouter":
            effective.append(entry)
            continue

        facts = index.get(name.lower())
        if facts is None:
            absent.append(name)
            if entry.get("enabled_by_default", True):
                withdrawn.append(name)
            entry["enabled_by_default"] = False
        else:
            entry.update(fact_fields(facts))
        effective.append(entry)

    added: list[str] = []
    seen: set[str] = set()
    for record in records:
        facts = openrouter_facts(record)
        model_id = facts["id"]
        keys = {model_id.lower(), model_id.split(":", 1)[0].lower()}
        slug = record.get("canonical_slug")
        if isinstance(slug, str):
            keys.add(slug.lower())
        if model_id in seen or keys & claimed or not _is_new_model_candidate(facts):
            continue
        seen.add(model_id)
        claimed.add(model_id.lower())
        effective.append(entry_from_facts(facts))
        added.append(model_id)

    return effective, {"withdrawn": withdrawn, "absent": absent, "added": added}


# ----------------------------------------------------------------------
# Cache and fetch
# ----------------------------------------------------------------------
def cache_path(filename: str) -> Path:
    return get_state_dir() / CACHE_SUBDIR / filename


def write_cache(filename: str, payload: Any, fetched_at: float) -> Path:
    """Write ``payload`` atomically (temp file plus rename) and return the path."""
    path = cache_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"fetched_at": fetched_at, "payload": payload}
    fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def read_cache(filename: str) -> tuple[Any, float] | None:
    """Return ``(payload, fetched_at)``, or ``None`` when absent or unreadable."""
    path = cache_path(filename)
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable registry cache %s: %s", path, exc)
        return None
    fetched_at = document.get("fetched_at") if isinstance(document, dict) else None
    if isinstance(fetched_at, bool) or not isinstance(fetched_at, (int, float)):
        logger.warning("Ignoring registry cache %s: no fetched_at timestamp", path)
        return None
    return document.get("payload"), float(fetched_at)


def fetch_json(url: str, timeout: float = FETCH_TIMEOUT_SECONDS) -> Any:
    """GET ``url`` and decode the JSON body. Raises on any network or decode error."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "pal-mcp-server"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read().decode("utf-8"))


def describe_age(fetched_at: float, now: float | None = None) -> str:
    seconds = max(0, int((now if now is not None else time.time()) - fetched_at))
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"
