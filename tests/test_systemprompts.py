"""
Baseline coverage for systemprompts/ modules.

For each prompt module we verify:

1. The module imports cleanly (catches syntax errors / missing imports).
2. The expected prompt constant is exported.
3. The constant is a non-empty string.
4. The prompt does not exceed a sanity upper bound on token estimate
   so a runaway prompt change is caught early.
5. Documented placeholder variables (``{var}`` style) match expected names.

The test deliberately does not snapshot prompt text — that would be far
too brittle for prose that is intentionally edited. Structural checks
catch the failure modes the issue calls out (typos, accidental newlines,
template-variable renames) without false positives on routine wording
tweaks.
"""

from __future__ import annotations

import importlib
import pkgutil
import re

import pytest

import systemprompts

# Mapping of module name -> expected exported constant name.
#
# Auto-derived: the convention in this package is
# ``<module_basename>_prompt.py`` -> ``<MODULE_BASENAME>_PROMPT`` with
# two documented exceptions:
#   * ``debug_prompt`` exports ``DEBUG_ISSUE_PROMPT`` (not ``DEBUG_PROMPT``).
#   * ``generate_code_prompt`` exports ``GENERATE_CODE_PROMPT``.
EXPECTED_EXPORTS: dict[str, str] = {
    "analyze_prompt": "ANALYZE_PROMPT",
    "chat_prompt": "CHAT_PROMPT",
    "codereview_prompt": "CODEREVIEW_PROMPT",
    "consensus_prompt": "CONSENSUS_PROMPT",
    "debug_prompt": "DEBUG_ISSUE_PROMPT",
    "docgen_prompt": "DOCGEN_PROMPT",
    "generate_code_prompt": "GENERATE_CODE_PROMPT",
    "jules_prompt": "JULES_PROMPT",
    "planner_prompt": "PLANNER_PROMPT",
    "precommit_prompt": "PRECOMMIT_PROMPT",
    "refactor_prompt": "REFACTOR_PROMPT",
    "secaudit_prompt": "SECAUDIT_PROMPT",
    "testgen_prompt": "TESTGEN_PROMPT",
    "thinkdeep_prompt": "THINKDEEP_PROMPT",
    "tracer_prompt": "TRACER_PROMPT",
}

# Placeholders that are intentional template variables consumed by the
# tool layer (e.g. ``str.format(stance_prompt=...)``). Any ``{name}``
# token in a prompt that is NOT listed here will trip the placeholder
# test — that's the early warning for accidental f-string-style debris
# or renamed variables.
EXPECTED_PLACEHOLDERS: dict[str, set[str]] = {
    "consensus_prompt": {"stance_prompt"},
}

# Upper bound on prompt size. ~4k tokens at a conservative
# 4-chars-per-token estimate = 16_000 chars. We allow a little slack
# to accommodate the larger structured prompts (refactor, secaudit,
# docgen) without inviting unbounded growth.
MAX_TOKEN_ESTIMATE = 6_000
MAX_CHARS = MAX_TOKEN_ESTIMATE * 4  # 24_000


def _discover_prompt_modules() -> list[str]:
    """Return the list of prompt module basenames under systemprompts/."""
    return sorted(
        name
        for _, name, ispkg in pkgutil.iter_modules(systemprompts.__path__)
        if not ispkg and name.endswith("_prompt")
    )


# ---------------------------------------------------------------------------
# Sanity check on the expected-exports table itself
# ---------------------------------------------------------------------------


def test_expected_exports_matches_filesystem() -> None:
    """The EXPECTED_EXPORTS table must enumerate every prompt module on disk.

    If this fails, a new prompt module was added (or one was renamed)
    without updating the test mapping — fix the mapping rather than
    weakening the assertion.
    """
    discovered = set(_discover_prompt_modules())
    declared = set(EXPECTED_EXPORTS)
    assert discovered == declared, (
        f"Prompt modules on disk ({sorted(discovered)}) do not match "
        f"EXPECTED_EXPORTS ({sorted(declared)}). "
        "Update tests/test_systemprompts.py::EXPECTED_EXPORTS."
    )


# ---------------------------------------------------------------------------
# Parametrised per-module checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("module_name", "export_name"), sorted(EXPECTED_EXPORTS.items()))
def test_prompt_module_imports_and_exports(module_name: str, export_name: str) -> None:
    """Every prompt module imports cleanly and exposes the expected constant."""
    module = importlib.import_module(f"systemprompts.{module_name}")
    assert hasattr(module, export_name), f"systemprompts.{module_name} is missing export {export_name!r}"


@pytest.mark.parametrize(("module_name", "export_name"), sorted(EXPECTED_EXPORTS.items()))
def test_prompt_is_nonempty_string(module_name: str, export_name: str) -> None:
    """Each exported prompt is a non-empty, non-whitespace string."""
    module = importlib.import_module(f"systemprompts.{module_name}")
    value = getattr(module, export_name)
    assert isinstance(value, str), f"{export_name} must be a str, got {type(value).__name__}"
    assert value.strip(), f"{export_name} is empty or whitespace-only"


@pytest.mark.parametrize(("module_name", "export_name"), sorted(EXPECTED_EXPORTS.items()))
def test_prompt_within_size_budget(module_name: str, export_name: str) -> None:
    """Catch runaway prompt growth via a conservative char/token upper bound."""
    module = importlib.import_module(f"systemprompts.{module_name}")
    value: str = getattr(module, export_name)
    estimated_tokens = len(value) // 4
    assert len(value) <= MAX_CHARS, (
        f"{export_name} is {len(value)} chars (~{estimated_tokens} tokens), "
        f"exceeding the {MAX_CHARS}-char ceiling. Either trim the prompt or, "
        f"if the growth is intentional, raise MAX_TOKEN_ESTIMATE in this test."
    )


@pytest.mark.parametrize(("module_name", "export_name"), sorted(EXPECTED_EXPORTS.items()))
def test_prompt_placeholders_match_expectations(module_name: str, export_name: str) -> None:
    """All ``{name}`` template variables match the documented set.

    Catches both *renamed* placeholders (``{stance_prompt}`` -> ``{stance}``)
    and *accidental* placeholders (an unescaped f-string debris token).
    """
    module = importlib.import_module(f"systemprompts.{module_name}")
    value: str = getattr(module, export_name)
    # Match {identifier} placeholders only. Skips JSON-shaped braces like
    # ``{"status": ...}`` because those don't match the ``\w+`` body
    # without commas/quotes.
    found = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", value))
    expected = EXPECTED_PLACEHOLDERS.get(module_name, set())
    assert found == expected, (
        f"{export_name} placeholder mismatch.\n"
        f"  expected: {sorted(expected)}\n"
        f"  found:    {sorted(found)}\n"
        "Update EXPECTED_PLACEHOLDERS if this is an intentional rename, "
        "or fix the prompt if a stray placeholder slipped in."
    )


# ---------------------------------------------------------------------------
# Cross-cutting checks against the package __init__
# ---------------------------------------------------------------------------


def test_package_reexports_every_prompt() -> None:
    """``systemprompts/__init__.py`` re-exports every per-module constant.

    Guards against a new prompt being added under systemprompts/ but
    not threaded through the package's public surface.
    """
    for module_name, export_name in EXPECTED_EXPORTS.items():
        assert hasattr(systemprompts, export_name), (
            f"systemprompts/__init__.py does not re-export {export_name} "
            f"from {module_name}. Add it to the imports and __all__."
        )
        # And the value re-exported is identical to the source constant.
        module = importlib.import_module(f"systemprompts.{module_name}")
        assert getattr(systemprompts, export_name) is getattr(module, export_name)


def test_package_all_matches_exports() -> None:
    """``systemprompts.__all__`` enumerates exactly the prompt constants."""
    declared = set(systemprompts.__all__)
    expected = set(EXPECTED_EXPORTS.values())
    assert declared == expected, (
        f"systemprompts.__all__ ({sorted(declared)}) does not match the "
        f"set of expected prompt constants ({sorted(expected)})."
    )
