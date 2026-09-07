"""
One definition of ``handle_completion_without_expert_analysis``, not two (issue #101).

The method used to exist twice: a concrete copy on ``BaseWorkflowMixin`` and another
on ``WorkflowTool``. Because ``WorkflowTool(BaseTool, BaseWorkflowMixin)`` puts
base.py first in the MRO, the mixin's copy was unreachable for every tool in the
registry — so the two drifted (the live copy keyed completion data with
``get_completion_data_key()``, the dead one with an f-string) and #96 had to write
the same behaviour change into both.

The mixin now declares the method abstract and ``WorkflowTool`` supplies the only
body. These tests pin that shape: a future edit that gives the mixin a concrete
implementation again re-creates a silently-dead second copy, and
``test_mixin_declares_the_hook_abstract`` is what fails when it does.
"""

from tools.workflow.base import WorkflowTool
from tools.workflow.workflow_mixin import BaseWorkflowMixin

METHOD = "handle_completion_without_expert_analysis"


class TestSingleDefinition:
    def test_mixin_declares_the_hook_abstract(self):
        """The mixin states the contract; it must not carry a body that nothing calls."""
        assert METHOD in BaseWorkflowMixin.__dict__, (
            f"{METHOD} disappeared from BaseWorkflowMixin. handle_work_completion calls it, "
            "so the contract belongs here even though WorkflowTool implements it."
        )

        declared = BaseWorkflowMixin.__dict__[METHOD]
        assert getattr(declared, "__isabstractmethod__", False), (
            f"BaseWorkflowMixin.{METHOD} has a concrete body again. WorkflowTool shadows it "
            "via the MRO, so that body is dead code that will silently drift from the live "
            "copy in tools/workflow/base.py (issue #101)."
        )
        assert METHOD in BaseWorkflowMixin.__abstractmethods__

    def test_workflow_tool_supplies_the_only_implementation(self):
        """The body every registered tool actually runs lives in tools/workflow/base.py."""
        assert METHOD in WorkflowTool.__dict__

        resolved = getattr(WorkflowTool, METHOD)
        assert resolved.__module__ == "tools.workflow.base", (
            f"{METHOD} resolved to {resolved.__module__}, not tools.workflow.base."
        )

        # WorkflowTool satisfies the abstract hook, so its subclasses stay instantiable.
        assert METHOD not in WorkflowTool.__abstractmethods__

    def test_mixin_is_the_only_place_the_hook_could_be_shadowed_from(self):
        """WorkflowTool is the sole mixin user, which is why the dead copy went unnoticed."""
        assert BaseWorkflowMixin.__subclasses__() == [WorkflowTool]
