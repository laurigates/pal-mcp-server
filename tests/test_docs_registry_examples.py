"""Guards model claims in docs/configuration.md and docs/custom_models.md against registry drift."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF = REPO_ROOT / "conf"
DOCS = REPO_ROOT / "docs"


def _models(registry: str) -> list[dict]:
    return json.loads((CONF / registry).read_text())["models"]


def _alias_map(registry: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for model in _models(registry):
        alias_map[model["model_name"].lower()] = model["model_name"]
        for alias in model.get("aliases", []):
            alias_map[alias.lower()] = model["model_name"]
    return alias_map


def test_opencode_go_row_counts_enabled_models():
    row = next(
        line
        for line in (DOCS / "configuration.md").read_text().splitlines()
        if line.strip().startswith("| OpenCode Go |")
    )
    match = re.search(r"\((\d+) models\)", row)
    assert match, f"OpenCode Go row no longer states a model count: {row}"

    enabled = sum(1 for model in _models("opencode_go_models.json") if model.get("enabled_by_default", True))
    assert int(match.group(1)) == enabled


def test_openrouter_alias_examples_resolve_as_documented():
    text = (DOCS / "custom_models.md").read_text()
    section = text.split("# OpenRouter models:", 1)[1].split("\n\n", 1)[0]
    examples = re.findall(r'"Use (\S+) [^"]*"\s+# → (\S+)', section)
    assert examples, "Expected OpenRouter alias examples in docs/custom_models.md"

    alias_map = _alias_map("openrouter_models.json")
    failures = [
        f"`{alias}` documented as {claimed}, registry resolves it to {alias_map.get(alias.lower())}"
        for alias, claimed in examples
        if alias_map.get(alias.lower()) != claimed
    ]
    assert not failures, "\n".join(failures)
