"""Regression tests for issue #200: prompt builders must emit real newlines.

Several workflow tools built their expert-analysis context, investigation
summaries and step guidance with ``"\\\\n"`` in the source, which is a
two-character backslash-n rather than a line break. The expert model then
received each section as one line with literal ``\\n`` text in it.
"""

import io
import tokenize
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.analyze import AnalyzeTool
from tools.codereview import CodeReviewTool
from tools.precommit import PrecommitTool
from tools.refactor import RefactorTool
from tools.shared.base_models import ConsolidatedFindings
from tools.testgen import TestGenTool
from tools.tracer import TracerTool

LITERAL_BACKSLASH_N = "\\n"


def _findings() -> ConsolidatedFindings:
    """Findings that populate every optional section of the expert context."""
    return ConsolidatedFindings(
        files_checked={"/src/a.py", "/src/b.py"},
        relevant_files={"/src/a.py"},
        relevant_context={"Foo.bar", "Foo.baz"},
        findings=["Step 1: looked at a.py", "Step 2: looked at b.py"],
        hypotheses=[
            {"step": 1, "confidence": "low", "hypothesis": "first idea"},
            {"step": 2, "confidence": "high", "hypothesis": "second idea"},
        ],
        issues_found=[
            {"severity": "high", "description": "bug one"},
            {"severity": "low", "description": "nit two"},
        ],
        images=["/img/one.png", "/img/two.png"],
    )


def _assert_real_newlines(text: str) -> None:
    assert LITERAL_BACKSLASH_N not in text
    assert "\n" in text


def _codereview() -> CodeReviewTool:
    tool = CodeReviewTool()
    tool.initial_request = "Review the auth module"
    tool.review_config = {"review_type": "full", "focus_on": "security"}
    return tool


def _analyze() -> AnalyzeTool:
    tool = AnalyzeTool()
    tool.initial_request = "Analyze the architecture"
    tool.analysis_config = {"analysis_type": "architecture", "output_format": "detailed"}
    return tool


def _precommit() -> PrecommitTool:
    tool = PrecommitTool()
    tool.initial_request = "Validate staged changes"
    tool.git_config = {"path": "/repo", "include_staged": True}
    return tool


def _refactor() -> RefactorTool:
    tool = RefactorTool()
    tool.initial_request = "Find code smells"
    tool.refactor_config = {"refactor_type": "codesmells", "style_guide_examples": None}
    return tool


def _testgen() -> TestGenTool:
    tool = TestGenTool()
    tool.initial_request = "Generate tests for Foo"
    return tool


TOOL_FACTORIES = {
    "codereview": _codereview,
    "analyze": _analyze,
    "precommit": _precommit,
    "refactor": _refactor,
    "testgen": _testgen,
}


@pytest.mark.parametrize("name", sorted(TOOL_FACTORIES))
def test_expert_analysis_context_uses_real_newlines(name):
    tool = TOOL_FACTORIES[name]()
    context = tool.prepare_expert_analysis_context(_findings())

    _assert_real_newlines(context)
    # Each banner starts its own line rather than being glued to the previous section.
    for line in context.splitlines():
        if "===" in line:
            assert line.startswith("==="), line


@pytest.mark.parametrize("name", sorted(TOOL_FACTORIES))
def test_work_summary_uses_real_newlines(name):
    tool = TOOL_FACTORIES[name]()
    tool.consolidated_findings = _findings()

    _assert_real_newlines(tool.prepare_work_summary())


def test_codereview_expert_context_lists_items_on_separate_lines():
    context = _codereview().prepare_expert_analysis_context(_findings())
    lines = context.splitlines()

    assert "=== CODE REVIEW REQUEST ===" in lines
    assert "Review the auth module" in lines
    assert "[HIGH] bug one" in lines
    assert "[LOW] nit two" in lines
    assert "- /img/one.png" in lines
    assert "Step 2 (high confidence): second idea" in lines


@pytest.mark.parametrize("confidence", ["exploring", "medium"])
def test_tracer_step_guidance_uses_real_newlines(confidence):
    tool = TracerTool()
    request = SimpleNamespace(step_number=2, total_steps=4, confidence=confidence, findings="some findings")

    response = tool.handle_work_continuation({}, request)

    _assert_real_newlines(response["next_steps"])


def test_tracer_ask_mode_prompt_uses_real_newlines():
    tool = TracerTool()
    tool.trace_config = {"trace_mode": "ask"}
    request = SimpleNamespace(step_number=1, total_steps=3, confidence="exploring", findings="start")

    response = tool.handle_work_continuation({}, request)

    _assert_real_newlines(response["next_steps"])


def _escaped_newline_literals(source: str) -> list[int]:
    """Line numbers of string literals whose source text spells ``\\\\n``.

    Only string tokens are inspected, so comments that mention the bug do not
    trip the guard. Docstrings and deliberate escapes (a regex, a JSON example)
    are string literals too and are still flagged: that is intentional, since
    tools/ has none today and a new one should be a conscious, reviewed choice.
    """
    string_types = {tokenize.STRING}
    if hasattr(tokenize, "FSTRING_MIDDLE"):  # Python 3.12+ splits f-strings into parts
        string_types.add(tokenize.FSTRING_MIDDLE)
    return [
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type in string_types and "\\\\n" in token.string
    ]


def test_escaped_newline_guard_detects_the_original_bug():
    buggy = 'header = "=== CODE REVIEW REQUEST ===\\\\n"\n# a comment about "\\\\n" is fine\n'
    fstring = 'line = f"{x}\\\\n"\n'

    assert _escaped_newline_literals(buggy) == [1]
    assert _escaped_newline_literals(fstring) == [1]
    # A single backslash in source is a real newline escape, not the bug.
    assert _escaped_newline_literals('ok = "a\\nb"\n') == []


def test_tool_sources_contain_no_escaped_newline_literals():
    """Guard against reintroducing ``"\\\\n"`` in any prompt builder under tools/."""
    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    offenders = [
        f"{path.relative_to(tools_dir.parent)}:{lineno}"
        for path in sorted(tools_dir.rglob("*.py"))
        for lineno in _escaped_newline_literals(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"Escaped newline literals found: {offenders}"
