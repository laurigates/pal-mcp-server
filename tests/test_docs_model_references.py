"""Guards docs/advanced-usage.md against drifting out of sync with conf/gemini_models.json.

Issue #139: the doc's Gemini alias table and per-provider capability list had gone
stale relative to the model registry (aliases pointed at generations the registry
no longer carries).

A fully general parser that could reliably associate *any* alias mention in prose
("Use flash to review formatting...") with a specific claimed model is not
tractable -- prose doesn't commit to a single, machine-checkable claim. Instead,
this doc uses one narrow, explicit, and easy-to-keep-honest markup wherever it
names a Gemini alias's resolved model:

    **`<alias>`** (resolves to `<model_name>`)

e.g. ``**`pro`** (resolves to `gemini-3.1-pro-preview`)``. This test parses every
occurrence of that pattern and checks it against conf/gemini_models.json:

1. The alias must actually be registered (case-insensitively) against some model.
2. The model named in the doc must be the same model the registry resolves that
   alias to.

This intentionally does not try to catch every stale mention of a model generation
in free-form prose elsewhere in the doc -- only the explicit annotation above is
checked. Any doc edit that introduces a new "(resolves to `...`)" annotation must
keep it accurate, and this test is the enforcement mechanism for that promise.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = REPO_ROOT / "docs" / "advanced-usage.md"
GEMINI_REGISTRY_PATH = REPO_ROOT / "conf" / "gemini_models.json"

# Matches: **`<alias>`** (resolves to `<model_name>`)
RESOLVES_TO_PATTERN = re.compile(r"\*\*`(?P<alias>[^`]+)`\*\*\s*\(resolves to `(?P<model>[^`]+)`\)")


def _load_gemini_alias_map() -> dict[str, str]:
    """Maps every lower-cased alias (and bare model name) to its canonical model_name."""
    data = json.loads(GEMINI_REGISTRY_PATH.read_text())
    alias_map: dict[str, str] = {}
    for model in data["models"]:
        model_name = model["model_name"]
        alias_map[model_name.lower()] = model_name
        for alias in model.get("aliases", []):
            alias_map[alias.lower()] = model_name
    return alias_map


def _find_doc_annotations() -> list[tuple[int, str, str]]:
    """Returns (line_number, alias, claimed_model) for every annotation in the doc."""
    lines = DOCS_PATH.read_text().splitlines()
    annotations = []
    for lineno, line in enumerate(lines, start=1):
        for match in RESOLVES_TO_PATTERN.finditer(line):
            annotations.append((lineno, match.group("alias"), match.group("model")))
    return annotations


def test_doc_contains_resolves_to_annotations():
    """Guards against the checkable annotation format silently disappearing from the doc.

    If this starts failing because the doc was rewritten without the
    "(resolves to `model`)" markup, either restore the markup or replace this
    whole test with an equivalent check against whatever markup replaced it.
    """
    assert _find_doc_annotations(), (
        "Expected docs/advanced-usage.md to contain at least one "
        "'**`<alias>`** (resolves to `<model>`)' annotation for a Gemini alias."
    )


def test_documented_gemini_aliases_match_registry():
    alias_map = _load_gemini_alias_map()
    annotations = [
        (lineno, alias, model) for lineno, alias, model in _find_doc_annotations() if model.startswith("gemini-")
    ]
    assert annotations, "Expected at least one documented alias resolving to a gemini-* model."

    failures = []
    for lineno, alias, claimed_model in annotations:
        actual_model = alias_map.get(alias.lower())
        if actual_model is None:
            failures.append(
                f"docs/advanced-usage.md:{lineno}: alias `{alias}` is documented as "
                f"resolving to `{claimed_model}`, but conf/gemini_models.json does not "
                f"register `{alias}` as an alias of any model."
            )
        elif actual_model != claimed_model:
            failures.append(
                f"docs/advanced-usage.md:{lineno}: alias `{alias}` is documented as "
                f"resolving to `{claimed_model}`, but conf/gemini_models.json resolves "
                f"it to `{actual_model}`."
            )

    assert not failures, "Doc/registry drift found:\n" + "\n".join(failures)
