"""
Model-context resolution for conversation continuation.

This module provides a single helper that encapsulates the fallback
cascade used by `reconstruct_thread_context` to pick an effective
`ModelContext` for rebuilding conversation history.

The cascade exists because `DEFAULT_MODEL=auto` and provider
availability mean the model named in the original request may not be
the one we can actually instantiate. Rather than re-deriving the
cascade inline three times (the original implementation did), we
funnel every entry point through `resolve_model_for_context` so the
fallback order is auditable in one place.

Cascade order:

1. If an `existing_model_context` is supplied (passed through via
   `arguments["_model_context"]`), trust it.
2. Otherwise try `ModelContext.from_arguments({"model": requested_model_name})`.
3. On `ValueError`, fall back to the tool's preferred fallback model
   (`ModelProviderRegistry.get_preferred_fallback_model(tool.get_model_category())`)
   when a tool is available.
4. If still nothing, fall back to the first entry in
   `ModelProviderRegistry.get_available_model_names()`.
5. As a final guard, if the resolved model has no provider available,
   re-run steps 3-4 to swap to a model that does.

If every step fails, a `ValueError` is raised so callers surface a
"no model available" error rather than crashing later on a None
provider.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from utils.model_context import ModelContext

if TYPE_CHECKING:
    from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


def _resolve_fallback_model(tool: BaseTool | None) -> str | None:
    """
    Pick the next model to try after the primary lookup failed.

    Prefers the tool's category-specific fallback; falls back to the
    first globally-available model name. Returns None when no models
    are available at all.
    """
    from providers.registry import ModelProviderRegistry

    fallback_model: str | None = None
    if tool is not None:
        try:
            fallback_model = ModelProviderRegistry.get_preferred_fallback_model(tool.get_model_category())
        except Exception as fallback_exc:  # pragma: no cover - defensive log
            logger.debug(
                "[CONVERSATION_DEBUG] Unable to resolve fallback model for "
                f"{getattr(tool, 'name', '<unknown>')}: {fallback_exc}"
            )

    if fallback_model is None:
        available_models = ModelProviderRegistry.get_available_model_names()
        if available_models:
            fallback_model = available_models[0]

    return fallback_model


def resolve_model_for_context(
    *,
    tool: BaseTool | None,
    requested_model_name: str | None = None,
    existing_model_context: Any = None,
) -> ModelContext:
    """
    Resolve a `ModelContext` for conversation reconstruction.

    Args:
        tool: The tool implementation whose conversation we're rebuilding.
            May be ``None`` if the tool name is unknown to the running
            server (older threads); the function then falls back to the
            globally-available model list rather than a per-category
            preference.
        requested_model_name: The model name from the inbound request
            (``arguments.get("model")``). May be ``None`` when the
            client did not request a specific model.
        existing_model_context: An already-constructed ``ModelContext``
            supplied via ``arguments.get("_model_context")``. Used
            directly when present.

    Returns:
        A ``ModelContext`` whose underlying model has a provider
        available with current credentials.

    Raises:
        ValueError: If no model can be resolved (no fallback, no
        available providers). The message points operators at the
        likely cause (missing API keys / invalid ``DEFAULT_MODEL``).
    """
    from providers.registry import ModelProviderRegistry

    # 1. Trust an explicit, caller-supplied ModelContext.
    if existing_model_context is not None:
        return existing_model_context

    # 2. Try to build a ModelContext from the requested model name.
    try:
        model_context = ModelContext.from_arguments({"model": requested_model_name} if requested_model_name else {})
    except ValueError as exc:
        fallback_model = _resolve_fallback_model(tool)
        if fallback_model is None:
            raise
        logger.debug(
            f"[CONVERSATION_DEBUG] Falling back to model '{fallback_model}' for context reconstruction after error: {exc}"
        )
        return ModelContext(fallback_model)

    # 3. Validate the resolved model actually has a provider available.
    #    `ModelContext.from_arguments` may succeed (it does not call the
    #    provider) only for the resolution to fail later when something
    #    accesses `model_context.provider`. Catch that here so we can
    #    swap to a fallback before returning.
    provider = ModelProviderRegistry.get_provider_for_model(model_context.model_name)
    if provider is None:
        fallback_model = _resolve_fallback_model(tool)
        if fallback_model is None:
            raise ValueError(
                f"Conversation continuation failed: model '{model_context.model_name}' is not available "
                f"with current API keys."
            )
        logger.debug(
            f"[CONVERSATION_DEBUG] Model '{model_context.model_name}' unavailable; swapping to "
            f"'{fallback_model}' for context reconstruction"
        )
        return ModelContext(fallback_model)

    return model_context
