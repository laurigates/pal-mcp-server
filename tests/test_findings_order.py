"""Workflow findings keep first-seen order, so expert prompts are byte-stable (issue #194, Stage A).

``files_checked``, ``relevant_files`` and ``relevant_context`` were sets. They are
rendered into expert prompts, and set iteration order depends on
``PYTHONHASHSEED``, so the same request produced different bytes in different
server processes and no provider cache could reuse it. Gate G4 in
``tests/test_prompt_prefix_stability.py`` checks the end-to-end property; these
tests pin the accumulation itself.
"""

from __future__ import annotations

from tools.thinkdeep import ThinkDeepTool
from tools.workflow.workflow_mixin import unique_in_order


def _step(number: int, **fields) -> dict:
    return {"step_number": number, "findings": f"step {number}", **fields}


def test_findings_lists_keep_first_seen_order_without_duplicates():
    workflow = ThinkDeepTool()

    workflow._update_consolidated_findings(
        _step(1, files_checked=["/z.py", "/a.py"], relevant_files=["/z.py"], relevant_context=["Z.run", "A.run"])
    )
    workflow._update_consolidated_findings(
        _step(
            2, files_checked=["/a.py", "/m.py"], relevant_files=["/m.py", "/z.py"], relevant_context=["A.run", "M.go"]
        )
    )

    findings = workflow.consolidated_findings
    assert findings.files_checked == ["/z.py", "/a.py", "/m.py"]
    assert findings.relevant_files == ["/z.py", "/m.py"]
    assert findings.relevant_context == ["Z.run", "A.run", "M.go"]


def test_images_are_deduplicated_in_first_seen_order():
    assert unique_in_order(["/b.png", "/a.png", "/b.png"]) == ["/b.png", "/a.png"]
