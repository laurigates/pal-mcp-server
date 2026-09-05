"""
Consensus tool - Step-by-step multi-model consensus with expert analysis

This tool provides a structured workflow for gathering consensus from multiple models.
It guides the CLI agent through systematic steps where the CLI agent first provides its own analysis,
then consults each requested model one by one, and finally synthesizes all perspectives.

Key features:
- Step-by-step consensus workflow with progress tracking
- The CLI agent's initial neutral analysis followed by model-specific consultations
- Context-aware file embedding
- Support for stance-based analysis (for/against/neutral)
- Final synthesis combining all perspectives
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from mcp.types import TextContent

from config import TEMPERATURE_ANALYTICAL
from systemprompts import CONSENSUS_PROMPT
from tools.shared.base_models import ConsolidatedFindings, WorkflowRequest
from utils.conversation_memory import MAX_CONVERSATION_TURNS, create_thread, get_thread
from utils.progress import get_progress_reporter, summarize_usage

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)

# Tool-specific field descriptions for consensus workflow
CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS = {
    "step": (
        "Step 1: the exact proposal every model sees, phrased as 'Evaluate…' (no meta commentary). "
        "Steps 2+: notes on the latest response; never sent to other models."
    ),
    "step_number": "Starts at 1. Step 1: your own analysis; steps 2+: one model response each.",
    "total_steps": "Models consulted plus one final synthesis step.",
    "next_step_required": "True while consultations remain; false when ready to synthesize.",
    "findings": (
        "Step 1: your own analysis, kept for synthesis (not shared with models). Steps 2+: summary of the newest "
        "response."
    ),
    "relevant_files": "Optional supporting files; absolute, non-abbreviated paths.",
    "models": (
        "Models to consult, at least two. Entry keys: model, stance (for/against/neutral), stance_prompt. (model, "
        "stance) pairs must be unique, e.g. [{'model':'gpt5','stance':'for'}, {'model':'pro','stance':'against'}]."
    ),
    "current_model_index": "0-based index of the next model to consult (managed internally).",
    "model_responses": "Internal log of responses gathered so far.",
    "images": "Optional absolute image paths or base64 references.",
}


class ConsensusRequest(WorkflowRequest):
    """Request model for consensus workflow steps"""

    # Required fields for each step
    step: str = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"])
    step_number: int = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"])
    total_steps: int = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"])
    next_step_required: bool = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"])

    # Investigation tracking fields
    findings: str = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"])
    confidence: str = Field(default="exploring", exclude=True, description="Not used")

    # Consensus-specific fields (only needed in step 1)
    models: list[dict] | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"])
    relevant_files: list[str] | None = Field(
        default_factory=list,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
    )

    # Internal tracking fields
    current_model_index: int | None = Field(
        0,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["current_model_index"],
    )
    model_responses: list[dict] | None = Field(
        default_factory=list,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["model_responses"],
    )

    # Optional images for visual debugging
    images: list[str] | None = Field(default=None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"])

    # Override inherited fields to exclude them from schema
    temperature: float | None = Field(default=None, exclude=True)
    thinking_mode: str | None = Field(default=None, exclude=True)

    # Not used in consensus workflow
    files_checked: list[str] | None = Field(default_factory=list, exclude=True)
    relevant_context: list[str] | None = Field(default_factory=list, exclude=True)
    issues_found: list[dict] | None = Field(default_factory=list, exclude=True)
    hypothesis: str | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_step_one_requirements(self):
        """Ensure step 1 has required models field and unique model+stance combinations."""
        if self.step_number == 1:
            if not self.models:
                raise ValueError("Step 1 requires 'models' field to specify which models to consult")

            # Check for unique model + stance combinations
            seen_combinations = set()
            for model_config in self.models:
                model_name = model_config.get("model", "")
                stance = model_config.get("stance", "neutral")
                combination = f"{model_name}:{stance}"

                if combination in seen_combinations:
                    raise ValueError(
                        f"Duplicate model + stance combination found: {model_name} with stance '{stance}'. "
                        f"Each model + stance combination must be unique."
                    )
                seen_combinations.add(combination)

        return self


class ConsensusTool(WorkflowTool):
    """
    Consensus workflow tool for step-by-step multi-model consensus gathering.

    This tool implements a structured consensus workflow where the CLI agent first provides
    its own neutral analysis, then consults each specified model individually,
    and finally synthesizes all perspectives into a unified recommendation.
    """

    def __init__(self):
        super().__init__()
        self.initial_prompt: str | None = None
        self.original_proposal: str | None = None  # Store the original proposal separately
        self.models_to_consult: list[dict] = []
        self.accumulated_responses: list[dict] = []
        self._current_arguments: dict[str, Any] = {}

    def get_name(self) -> str:
        return "consensus"

    def get_description(self) -> str:
        return (
            "Sends one proposal to several models with for/against/neutral stances; returns each answer for you to "
            "synthesize. Use for design or technology decisions needing independent views. Not for decisions with one "
            "clear answer."
        )

    def get_system_prompt(self) -> str:
        # For the CLI agent's initial analysis, use a neutral version of the consensus prompt
        return CONSENSUS_PROMPT.replace(
            "{stance_prompt}",
            """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        )

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> ToolModelCategory:
        """Consensus workflow requires extended reasoning"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for consensus workflow."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        # Consensus tool-specific field definitions
        consensus_field_overrides = {
            # Override standard workflow fields that need consensus-specific descriptions
            "step": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"],
            },
            "step_number": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"],
            },
            "total_steps": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"],
            },
            "next_step_required": {
                "type": "boolean",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"],
            },
            "findings": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"],
            },
            "relevant_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
            },
            # consensus-specific fields (not in base workflow)
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "stance": {"type": "string", "enum": ["for", "against", "neutral"], "default": "neutral"},
                        "stance_prompt": {"type": "string"},
                    },
                    "required": ["model"],
                },
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"],
                "minItems": 2,
            },
            "current_model_index": {
                "type": "integer",
                "minimum": 0,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["current_model_index"],
            },
            "model_responses": {
                "type": "array",
                "items": {"type": "object"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["model_responses"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"],
            },
            "use_assistant_model": {
                "type": "boolean",
                "default": True,
                "description": "Ignored: consensus runs no expert analysis.",
            },
        }

        # Provide guidance on available models similar to single-model tools
        model_description = (
            "If the user names a model, use that exact value or report the provider error; never substitute. "
            "`listmodels` lists the roster."
        )

        summaries, total, restricted = self._get_ranked_model_summaries()
        remainder = max(0, total - len(summaries))
        if summaries:
            label = "Allowed models" if restricted else "Top models"
            top_line = "; ".join(summaries)
            if remainder > 0:
                top_line = f"{label}: {top_line}; +{remainder} more via `listmodels`."
            else:
                top_line = f"{label}: {top_line}."
            model_description = f"{model_description} {top_line}"
        else:
            model_description = (
                f"{model_description} No models detected; configure provider credentials or check `listmodels`."
            )

        restriction_note = self._get_restriction_note()
        if restriction_note and (remainder > 0 or not summaries):
            model_description = f"{model_description} {restriction_note}."

        existing_models_desc = consensus_field_overrides["models"]["description"]
        consensus_field_overrides["models"]["description"] = f"{existing_models_desc} {model_description}"

        # Define excluded fields for consensus workflow
        excluded_workflow_fields = [
            "files_checked",  # Not used in consensus workflow
            "relevant_context",  # Not used in consensus workflow
            "issues_found",  # Not used in consensus workflow
            "hypothesis",  # Not used in consensus workflow
            "confidence",  # Not used in consensus workflow
        ]

        excluded_common_fields = [
            "model",  # Consensus uses 'models' field instead
            "temperature",  # Not used in consensus workflow
            "thinking_mode",  # Not used in consensus workflow
        ]

        requires_model = self.requires_model()
        model_field_schema = self.get_model_field_schema() if requires_model else None
        auto_mode = self.is_effective_auto_mode() if requires_model else False

        return WorkflowSchemaBuilder.build_schema(
            tool_specific_fields=consensus_field_overrides,
            model_field_schema=model_field_schema,
            auto_mode=auto_mode,
            tool_name=self.get_name(),
            excluded_workflow_fields=excluded_workflow_fields,
            excluded_common_fields=excluded_common_fields,
            require_model=requires_model,
        )

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:  # noqa: ARG002
        """Define required actions for each consensus phase.

        Now includes request parameter for continuation-aware decisions.
        Note: confidence parameter is kept for compatibility with base class but not used.
        """
        if step_number == 1:
            # CLI Agent's initial analysis
            return [
                "You've provided your initial analysis. The tool will now consult other models.",
                "Wait for the next step to receive the first model's response.",
            ]
        elif step_number < total_steps - 1:
            # Processing individual model responses
            return [
                "Review the model response provided in this step",
                "Note key agreements and disagreements with previous analyses",
                "Wait for the next model's response",
            ]
        else:
            # Ready for final synthesis
            return [
                "All models have been consulted",
                "Synthesize all perspectives into a comprehensive recommendation",
                "Identify key points of agreement and disagreement",
                "Provide clear, actionable guidance based on the consensus",
            ]

    def should_call_expert_analysis(self, consolidated_findings, request=None) -> bool:
        """Consensus workflow doesn't use traditional expert analysis - it consults models step by step."""
        return False

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """Not used in consensus workflow."""
        return ""

    def requires_expert_analysis(self) -> bool:
        """Consensus workflow handles its own model consultations."""
        return False

    def requires_model(self) -> bool:
        """
        Consensus tool doesn't require model resolution at the MCP boundary.

        Uses it's own set of models

        Returns:
            bool: False
        """
        return False

    # Hook method overrides for consensus-specific behavior

    def prepare_step_data(self, request) -> dict:
        """Prepare consensus-specific step data."""
        step_data = {
            "step": request.step,
            "step_number": request.step_number,
            "findings": request.findings,
            "files_checked": [],  # Not used
            "relevant_files": request.relevant_files or [],
            "relevant_context": [],  # Not used
            "issues_found": [],  # Not used
            "confidence": "exploring",  # Not used, kept for compatibility
            "hypothesis": None,  # Not used
            "images": request.images or [],  # Now used for visual context
        }
        return step_data

    def _build_complete_consensus(self) -> dict:
        """Summarise the finished roster, making any lost models impossible to miss.

        A model that fails at call time (retired model id, provider outage, bad key)
        is recorded with status "error" and the workflow carries on. Counting those
        as responses reported a full-strength consensus that never happened, so the
        summary separates what answered from what did not.
        """
        succeeded = [m for m in self.accumulated_responses if m.get("status") == "success"]
        failed = [m for m in self.accumulated_responses if m.get("status") != "success"]

        summary = {
            "initial_prompt": self.original_proposal if self.original_proposal else self.initial_prompt,
            "models_consulted": [f"{m['model']}:{m.get('stance', 'neutral')}" for m in succeeded],
            "total_responses": len(succeeded),
            "models_requested": len(self.accumulated_responses),
            "consensus_confidence": "high" if not failed else "degraded",
        }

        if failed:
            summary["models_failed"] = [
                {
                    "model": m["model"],
                    "stance": m.get("stance", "neutral"),
                    "error": m.get("error", "unknown error"),
                }
                for m in failed
            ]
            # Provider coverage matters independently of the count: losing the only
            # model from a given provider removes a whole perspective, not just a vote.
            answered_providers = {m.get("metadata", {}).get("provider") for m in succeeded}
            answered_providers.discard(None)
            summary["providers_represented"] = sorted(answered_providers)

        return summary

    def _synthesis_next_steps(self, consensus: dict) -> str:
        """Build the synthesis instruction, leading with a warning when degraded."""
        preamble = ""
        if consensus.get("consensus_confidence") == "degraded":
            failed = consensus.get("models_failed", [])
            names = ", ".join(f"{f['model']}" for f in failed)
            preamble = (
                f"⚠️ CONSENSUS DEGRADED: {consensus['total_responses']} of "
                f"{consensus['models_requested']} models answered. Failed: {names}. "
                "You MUST state this limitation in your synthesis — the result is weaker "
                "than the consensus that was requested, and provider coverage may be lost.\n\n"
            )

        return preamble + (
            "CONSENSUS GATHERING IS COMPLETE. You MUST now synthesize all perspectives and present:\n"
            "1. Key points of AGREEMENT across models\n"
            "2. Key points of DISAGREEMENT and why they differ\n"
            "3. Your final consolidated recommendation\n"
            "4. Specific, actionable next steps for implementation\n"
            "5. Critical risks or concerns that must be addressed"
        )

    async def handle_work_completion(self, response_data: dict, request, arguments: dict) -> dict:  # noqa: ARG002
        """Handle consensus workflow completion - no expert analysis, just final synthesis."""
        response_data["consensus_complete"] = True
        response_data["status"] = "consensus_workflow_complete"

        # Prepare final synthesis data
        consensus = self._build_complete_consensus()
        response_data["complete_consensus"] = consensus
        response_data["next_steps"] = self._synthesis_next_steps(consensus)

        return response_data

    def handle_work_continuation(self, response_data: dict, request) -> dict:
        """Handle continuation between consensus steps."""
        current_idx = request.current_model_index or 0

        if request.step_number == 1:
            # After CLI Agent's initial analysis, prepare to consult first model
            response_data["status"] = "consulting_models"
            response_data["next_model"] = self.models_to_consult[0] if self.models_to_consult else None
            response_data["next_steps"] = (
                "Your initial analysis is complete. The tool will now consult the specified models."
            )
        elif current_idx < len(self.models_to_consult):
            next_model = self.models_to_consult[current_idx]
            response_data["status"] = "consulting_next_model"
            response_data["next_model"] = next_model
            response_data["models_remaining"] = len(self.models_to_consult) - current_idx
            response_data["next_steps"] = f"Model consultation in progress. Next: {next_model['model']}"
        else:
            response_data["status"] = "ready_for_synthesis"
            response_data["next_steps"] = "All models consulted. Ready for final synthesis."

        return response_data

    async def execute_workflow(self, arguments: dict[str, Any]) -> list:
        """Override execute_workflow to handle model consultations between steps."""

        # Store arguments
        self._current_arguments = arguments

        # Validate request
        request = self.get_workflow_request_model()(**arguments)

        # Resolve existing continuation_id or create a new one on first step
        continuation_id = request.continuation_id

        if request.step_number == 1:
            if not continuation_id:
                clean_args = {
                    k: v
                    for k, v in arguments.items()
                    if k
                    not in ["_model_context", "_resolved_model_name", "_expert_model_called", "_expert_provider_called"]
                }
                continuation_id = create_thread(self.get_name(), clean_args)
                request.continuation_id = continuation_id
                arguments["continuation_id"] = continuation_id
                self.work_history = []
                self.consolidated_findings = ConsolidatedFindings()

            # Store the original proposal from step 1 - this is what all models should see
            self.store_initial_issue(request.step)
            self.initial_request = request.step
            self.models_to_consult = request.models or []
            self.accumulated_responses = []
            # Set total steps: len(models) (each step includes consultation + response)
            request.total_steps = len(self.models_to_consult)

        # For all steps (1 through total_steps), consult the corresponding model
        if request.step_number <= request.total_steps:
            # Calculate which model to consult for this step
            model_idx = request.step_number - 1  # 0-based index

            if model_idx < len(self.models_to_consult):
                # Track workflow state for conversation memory
                step_data = self.prepare_step_data(request)
                self.work_history.append(step_data)
                self._update_consolidated_findings(step_data)

                # Consult the model for this step
                model_response = await self._consult_model(self.models_to_consult[model_idx], request)

                # Add to accumulated responses
                self.accumulated_responses.append(model_response)

                # Include the model response in the step data
                response_data = {
                    "status": "model_consulted",
                    "step_number": request.step_number,
                    "total_steps": request.total_steps,
                    "model_consulted": model_response["model"],
                    "model_stance": model_response.get("stance", "neutral"),
                    "model_response": model_response,
                    "current_model_index": model_idx + 1,
                    "next_step_required": request.step_number < request.total_steps,
                }

                # Add CLAI Agent's analysis to step 1
                if request.step_number == 1:
                    response_data["agent_analysis"] = {
                        "initial_analysis": request.step,
                        "findings": request.findings,
                    }
                    response_data["status"] = "analysis_and_first_model_consulted"

                # Check if this is the final step
                if request.step_number == request.total_steps:
                    response_data["status"] = "consensus_workflow_complete"
                    response_data["consensus_complete"] = True
                    consensus = self._build_complete_consensus()
                    response_data["complete_consensus"] = consensus
                    response_data["next_steps"] = self._synthesis_next_steps(consensus)
                else:
                    response_data["next_steps"] = (
                        f"Model {model_response['model']} has provided its {model_response.get('stance', 'neutral')} "
                        f"perspective. Please analyze this response and call {self.get_name()} again with:\n"
                        f"- step_number: {request.step_number + 1}\n"
                        f"- findings: Summarize key points from this model's response"
                    )

                # Add continuation information and workflow customization
                response_data = self.customize_workflow_response(response_data, request)

                # Ensure consensus-specific metadata is attached
                self._add_workflow_metadata(response_data, arguments)

                if continuation_id:
                    self.store_conversation_turn(continuation_id, response_data, request)
                    continuation_offer = self._build_continuation_offer(continuation_id)
                    if continuation_offer:
                        response_data["continuation_offer"] = continuation_offer

                return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

        # Otherwise, use standard workflow execution
        return await super().execute_workflow(arguments)

    def _build_continuation_offer(self, continuation_id: str) -> dict[str, Any] | None:
        """Create a continuation offer without exposing prior model responses."""
        try:
            from tools.models import ContinuationOffer

            thread = get_thread(continuation_id)
            if thread and thread.turns:
                remaining_turns = max(0, MAX_CONVERSATION_TURNS - len(thread.turns))
            else:
                remaining_turns = MAX_CONVERSATION_TURNS - 1

            # Provide a neutral note specific to consensus workflow
            note = (
                f"Consensus workflow can continue for {remaining_turns} more exchanges."
                if remaining_turns > 0
                else "Consensus workflow continuation limit reached."
            )

            continuation_offer = ContinuationOffer(
                continuation_id=continuation_id,
                note=note,
                remaining_turns=remaining_turns,
            )
            return continuation_offer.model_dump()
        except Exception:
            return None

    async def _consult_model(self, model_config: dict, request) -> dict:
        """Consult a single model and return its response."""
        try:
            # Import and create ModelContext once at the beginning
            from utils.model_context import ModelContext

            # Get the provider for this model
            model_name = model_config["model"]
            provider = self.get_model_provider(model_name)

            # Create model context once and reuse for both file processing and temperature validation
            model_context = ModelContext(model_name=model_name)

            # Prepare the prompt with any relevant files
            # Use continuation_id=None for blinded consensus - each model should only see
            # original prompt + files, not conversation history or other model responses
            # CRITICAL: Use the original proposal from step 1, NOT what's in request.step for steps 2+!
            # Steps 2+ contain summaries/notes that must NEVER be sent to other models
            prompt = self.original_proposal if self.original_proposal else self.initial_prompt
            if request.relevant_files:
                file_content, _ = self._prepare_file_content_for_prompt(
                    request.relevant_files,
                    None,  # Use None instead of request.continuation_id for blinded consensus
                    "Context files",
                    model_context=model_context,
                )
                if file_content:
                    prompt = f"{prompt}\n\n=== CONTEXT FILES ===\n{file_content}\n=== END CONTEXT ==="

            # Get stance-specific system prompt
            stance = model_config.get("stance", "neutral")
            stance_prompt = model_config.get("stance_prompt")
            system_prompt = self._get_stance_enhanced_prompt(stance, stance_prompt)

            # Validate temperature against model constraints (respects supports_temperature)
            validated_temperature, temp_warnings = self.validate_and_correct_temperature(
                self.get_default_temperature(), model_context
            )

            # Log any temperature corrections
            for warning in temp_warnings:
                logger.warning(warning)

            # Call the model with validated temperature. Each consensus step
            # consults exactly one model, so name it and where it sits in the
            # roster — otherwise a three-model consensus is three silent minutes.
            progress = get_progress_reporter()
            position = f"{request.step_number} of {request.total_steps}"
            stance_label = f" ({stance})" if stance and stance != "neutral" else ""
            async with progress.heartbeat(f"consensus · consulting {model_name}{stance_label} · {position}"):
                response = await provider.generate_content(
                    prompt=prompt,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    temperature=validated_temperature,
                    thinking_mode="medium",
                    images=request.images if request.images else None,
                )
            await progress.update(f"consensus · {model_name} replied · {summarize_usage(response.usage)} · {position}")

            return {
                "model": model_name,
                "stance": stance,
                "status": "success",
                "verdict": response.content,
                "metadata": {
                    "provider": provider.get_provider_type().value,
                    "model_name": model_name,
                },
            }

        except Exception as e:
            logger.exception("Error consulting model %s", model_config)
            return {
                "model": model_config.get("model", "unknown"),
                "stance": model_config.get("stance", "neutral"),
                "status": "error",
                "error": str(e),
            }

    def _get_stance_enhanced_prompt(self, stance: str, custom_stance_prompt: str | None = None) -> str:
        """Get the system prompt with stance injection."""
        base_prompt = CONSENSUS_PROMPT

        if custom_stance_prompt:
            return base_prompt.replace("{stance_prompt}", custom_stance_prompt)

        stance_prompts = {
            "for": """SUPPORTIVE PERSPECTIVE WITH INTEGRITY

You are tasked with advocating FOR this proposal, but with CRITICAL GUARDRAILS:

MANDATORY ETHICAL CONSTRAINTS:
- This is NOT a debate for entertainment. You MUST act in good faith and in the best interest of the questioner
- You MUST think deeply about whether supporting this idea is safe, sound, and passes essential requirements
- You MUST be direct and unequivocal in saying "this is a bad idea" when it truly is
- There must be at least ONE COMPELLING reason to be optimistic, otherwise DO NOT support it

WHEN TO REFUSE SUPPORT (MUST OVERRIDE STANCE):
- If the idea is fundamentally harmful to users, project, or stakeholders
- If implementation would violate security, privacy, or ethical standards
- If the proposal is technically infeasible within realistic constraints
- If costs/risks dramatically outweigh any potential benefits

YOUR SUPPORTIVE ANALYSIS SHOULD:
- Identify genuine strengths and opportunities
- Propose solutions to overcome legitimate challenges
- Highlight synergies with existing systems
- Suggest optimizations that enhance value
- Present realistic implementation pathways

Remember: Being "for" means finding the BEST possible version of the idea IF it has merit, not blindly supporting bad ideas.""",
            "against": """CRITICAL PERSPECTIVE WITH RESPONSIBILITY

You are tasked with critiquing this proposal, but with ESSENTIAL BOUNDARIES:

MANDATORY FAIRNESS CONSTRAINTS:
- You MUST NOT oppose genuinely excellent, common-sense ideas just to be contrarian
- You MUST acknowledge when a proposal is fundamentally sound and well-conceived
- You CANNOT give harmful advice or recommend against beneficial changes
- If the idea is outstanding, say so clearly while offering constructive refinements

WHEN TO MODERATE CRITICISM (MUST OVERRIDE STANCE):
- If the proposal addresses critical user needs effectively
- If it follows established best practices with good reason
- If benefits clearly and substantially outweigh risks
- If it's the obvious right solution to the problem

YOUR CRITICAL ANALYSIS SHOULD:
- Identify legitimate risks and failure modes
- Point out overlooked complexities
- Suggest more efficient alternatives
- Highlight potential negative consequences
- Question assumptions that may be flawed

Remember: Being "against" means rigorous scrutiny to ensure quality, not undermining good ideas that deserve support.""",
            "neutral": """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        }

        stance_prompt = stance_prompts.get(stance, stance_prompts["neutral"])
        return base_prompt.replace("{stance_prompt}", stance_prompt)

    def customize_workflow_response(self, response_data: dict, request) -> dict:
        """Customize response for consensus workflow."""
        # Store model responses in the response for tracking
        if self.accumulated_responses:
            response_data["accumulated_responses"] = self.accumulated_responses

        # Add consensus-specific fields
        if request.step_number == 1:
            response_data["consensus_workflow_status"] = "initial_analysis_complete"
        elif request.step_number < request.total_steps - 1:
            response_data["consensus_workflow_status"] = "consulting_models"
        else:
            response_data["consensus_workflow_status"] = "ready_for_synthesis"

        # Customize metadata for consensus workflow
        self._customize_consensus_metadata(response_data, request)

        return response_data

    def _customize_consensus_metadata(self, response_data: dict, request) -> None:
        """
        Customize metadata for consensus workflow to accurately reflect multi-model nature.

        The default workflow metadata shows the model running Agent's analysis steps,
        but consensus is a multi-model tool that consults different models. We need
        to provide accurate metadata that reflects this.
        """
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        metadata = response_data["metadata"]

        # Always preserve tool_name
        metadata["tool_name"] = self.get_name()

        if request.step_number == request.total_steps:
            # Final step - show comprehensive consensus metadata
            models_consulted = []
            if self.models_to_consult:
                models_consulted = [f"{m['model']}:{m.get('stance', 'neutral')}" for m in self.models_to_consult]

            metadata.update(
                {
                    "workflow_type": "multi_model_consensus",
                    "models_consulted": models_consulted,
                    "consensus_complete": True,
                    "total_models": len(self.models_to_consult) if self.models_to_consult else 0,
                }
            )

            # Remove the misleading single model metadata
            metadata.pop("model_used", None)
            metadata.pop("provider_used", None)

        else:
            # Intermediate steps - show consensus workflow in progress
            models_to_consult = []
            if self.models_to_consult:
                models_to_consult = [f"{m['model']}:{m.get('stance', 'neutral')}" for m in self.models_to_consult]

            metadata.update(
                {
                    "workflow_type": "multi_model_consensus",
                    "models_to_consult": models_to_consult,
                    "consultation_step": request.step_number,
                    "total_consultation_steps": request.total_steps,
                }
            )

            # Remove the misleading single model metadata that shows Agent's execution model
            # instead of the models being consulted
            metadata.pop("model_used", None)
            metadata.pop("provider_used", None)

    def _add_workflow_metadata(self, response_data: dict, arguments: dict[str, Any]) -> None:
        """
        Override workflow metadata addition for consensus tool.

        The consensus tool doesn't use single model metadata because it's a multi-model
        workflow. Instead, we provide consensus-specific metadata that accurately
        reflects the models being consulted.
        """
        # Initialize metadata if not present
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        # Add basic tool metadata
        response_data["metadata"]["tool_name"] = self.get_name()

        # The consensus-specific metadata is already added by _customize_consensus_metadata
        # which is called from customize_workflow_response. We don't add the standard
        # single-model metadata (model_used, provider_used) because it's misleading
        # for a multi-model consensus workflow.

        logger.debug(
            f"[CONSENSUS_METADATA] {self.get_name()}: Using consensus-specific metadata instead of single-model metadata"
        )

    def store_initial_issue(self, step_description: str):
        """Store initial prompt for model consultations."""
        self.original_proposal = step_description
        self.initial_prompt = step_description  # Keep for backward compatibility

    # Required abstract methods from BaseTool
    def get_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    async def prepare_prompt(self, request) -> str:  # noqa: ARG002
        """Not used - workflow tools use execute_workflow()."""
        return ""  # Workflow tools use execute_workflow() directly
