"""
Model Restriction Service

This module provides centralized management of model usage restrictions
based on environment variables. It allows organizations to limit which
models can be used from each provider for cost control, compliance, or
standardization purposes.

Environment Variables:
    Every registered provider has one, and they are NOT enumerated here -- an
    enumeration in this file is exactly the drift that made three providers
    silently unrestrictable. Each provider class declares its own via
    ``ALLOWED_MODELS_ENV``, defaulting to
    ``f"{provider_type.value.upper()}_ALLOWED_MODELS"``. Ask the code:
    ``ModelProviderRegistry.allowed_models_env_vars(provider_type)``.

    An unset or empty value means "no restriction", not "allow nothing".

Example:
    OPENAI_ALLOWED_MODELS=o3-mini,o4-mini
    GOOGLE_ALLOWED_MODELS=flash
    XAI_ALLOWED_MODELS=grok-4.6,grok-4.3
    OPENROUTER_ALLOWED_MODELS=opus,sonnet,mistral
"""

import logging
from collections import defaultdict

from providers.shared import ProviderType
from utils.env import get_env

logger = logging.getLogger(__name__)


class ModelRestrictionService:
    """Central authority for environment-driven model allowlists.

    Role
        Interpret ``*_ALLOWED_MODELS`` environment variables, keep their
        entries normalised (lowercase), and answer whether a provider/model
        pairing is permitted.

    Responsibilities
        * Parse, cache, and expose per-provider restriction sets
        * Validate configuration by cross-checking each entry against the
          provider’s alias-aware model list
        * Offer helper methods such as ``is_allowed`` and ``filter_models`` to
          enforce policy everywhere model names appear (tool selection, CLI
          commands, etc.).
    """

    @staticmethod
    def env_vars_for(provider_type: ProviderType) -> tuple[str, ...]:
        """Allow-list env vars for a provider, most specific first."""
        try:
            from providers.registry import ModelProviderRegistry

            return ModelProviderRegistry.allowed_models_env_vars(provider_type)
        except (ImportError, AttributeError, NameError) as exc:
            # Policy must never fail to load, but it must not degrade silently
            # either: for AZURE the fallback drops AZURE_OPENAI_ALLOWED_MODELS
            # -- the documented name -- and the service is a process-lifetime
            # singleton, so one unlucky early call would bake that in.
            fallback = f"{provider_type.value.upper()}_ALLOWED_MODELS"
            logger.warning(
                "Could not resolve the allow-list variable for %s from the provider registry (%s); "
                "falling back to %s. A provider-declared override would be ignored.",
                provider_type.value,
                exc,
                fallback,
            )
            return (fallback,)

    def __init__(self):
        """Initialize the restriction service by loading from environment."""
        self.restrictions: dict[ProviderType, set[str]] = {}
        self._env_var_used: dict[ProviderType, str] = {}
        self._alias_resolution_cache: dict[ProviderType, dict[str, str]] = defaultdict(dict)
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load restrictions for every provider type."""
        for provider_type in ProviderType:
            env_vars = self.env_vars_for(provider_type)
            for env_var in env_vars:
                env_value = get_env(env_var)

                if env_value is None or env_value == "":
                    continue

                models = set()
                for model in env_value.split(","):
                    cleaned = model.strip().lower()
                    if cleaned:
                        models.add(cleaned)

                if models:
                    self.restrictions[provider_type] = models
                    self._env_var_used[provider_type] = env_var
                    self._alias_resolution_cache[provider_type] = {}
                    logger.info(f"{provider_type.value} allowed models: {sorted(models)}")
                    break
                logger.debug(f"{env_var} contains only whitespace - all {provider_type.value} models allowed")
            else:
                logger.debug(f"No allow-list set for {provider_type.value} - all models allowed")

    def validate_against_known_models(self, provider_instances: dict[ProviderType, any]) -> None:
        """
        Validate restrictions against known models from providers.

        This should be called after providers are initialized to warn about
        typos or invalid model names in the restriction lists.

        Note:
            ``provider.validate_model_name`` routes through
            ``_ensure_model_allowed``, which consults the *module singleton*
            restriction service rather than ``self``. In production they are
            the same object (``server.py`` validates the service it just
            fetched), so this is correct there -- but a caller holding a
            locally-constructed service must patch
            ``utils.model_restrictions._restriction_service`` for the results
            to mean anything.

        Args:
            provider_instances: Dictionary of provider type to provider instance
        """
        for provider_type, allowed_models in self.restrictions.items():
            provider = provider_instances.get(provider_type)
            if not provider:
                continue

            # Ask the provider whether it would accept each name, rather than
            # set-comparing against its manifest. The manifest is authoritative
            # for most providers, but OpenRouter's `_lookup_capabilities`
            # fabricates capabilities for any "vendor/model" name, so a valid
            # entry absent from conf/openrouter_models.json is not a typo.
            # Asking the provider gets both right, and still catches a typo'd
            # OpenRouter *alias* ("opuss"), which a provider-wide skip would not.
            # Built unconditionally rather than only when a warning fires: three
            # tests pin that validation goes through this alias-aware
            # polymorphic call, and one capability walk per restricted provider
            # at startup is not worth loosening that contract for.
            try:
                known_models = sorted(
                    provider.list_models(
                        respect_restrictions=False,
                        include_aliases=True,
                        lowercase=True,
                        unique=True,
                    )
                )
            except Exception as e:  # pragma: no cover - diagnostic aid only
                logger.debug(f"Could not get model list from {provider_type.value} provider: {e}")
                known_models = []

            env_var = self._env_var_used.get(provider_type, self.env_vars_for(provider_type)[0])
            # sorted() rather than iterating the set directly: is_allowed() can
            # add to self.restrictions[provider_type] while resolving aliases,
            # which would be a "Set changed size during iteration" away from a
            # crash. Also makes warning order deterministic.
            for allowed_model in sorted(allowed_models):
                try:
                    recognized = provider.validate_model_name(allowed_model)
                except Exception as e:  # pragma: no cover - never block startup on this check
                    logger.debug(f"Could not validate '{allowed_model}' against {provider_type.value}: {e}")
                    continue

                if not recognized:
                    logger.warning(
                        f"Model '{allowed_model}' in {env_var} "
                        f"is not a recognized {provider_type.value} model. "
                        f"Please check for typos. Known models: {known_models}"
                    )

    def is_allowed(self, provider_type: ProviderType, model_name: str, original_name: str | None = None) -> bool:
        """
        Check if a model is allowed for a specific provider.

        Args:
            provider_type: The provider type (OPENAI, GOOGLE, etc.)
            model_name: The canonical model name (after alias resolution)
            original_name: The original model name before alias resolution (optional)

        Returns:
            True if allowed (or no restrictions), False if restricted
        """
        if provider_type not in self.restrictions:
            # No restrictions for this provider
            return True

        allowed_set = self.restrictions[provider_type]

        if len(allowed_set) == 0:
            # Empty set - allowed
            return True

        # Check both the resolved name and original name (if different)
        names_to_check = {model_name.lower()}
        if original_name and original_name.lower() != model_name.lower():
            names_to_check.add(original_name.lower())

        # If any of the names is in the allowed set, it's allowed
        if any(name in allowed_set for name in names_to_check):
            return True

        # Attempt to resolve canonical names for allowed aliases using provider metadata.
        try:
            from providers.registry import ModelProviderRegistry

            provider = ModelProviderRegistry.get_provider(provider_type)
        except Exception:  # pragma: no cover - registry lookup failure shouldn't break validation
            provider = None

        if provider:
            cache = self._alias_resolution_cache.setdefault(provider_type, {})

            for allowed_entry in list(allowed_set):
                normalized_resolved = cache.get(allowed_entry)

                if not normalized_resolved:
                    try:
                        resolved = provider._resolve_model_name(allowed_entry)
                    except Exception:  # pragma: no cover - resolution failures are treated as non-matches
                        continue

                    if not resolved:
                        continue

                    normalized_resolved = resolved.lower()
                    cache[allowed_entry] = normalized_resolved

                if normalized_resolved in names_to_check:
                    allowed_set.add(normalized_resolved)
                    cache[normalized_resolved] = normalized_resolved
                    return True

        return False

    def get_allowed_models(self, provider_type: ProviderType) -> set[str] | None:
        """
        Get the set of allowed models for a provider.

        Args:
            provider_type: The provider type

        Returns:
            Set of allowed model names, or None if no restrictions
        """
        return self.restrictions.get(provider_type)

    def has_restrictions(self, provider_type: ProviderType) -> bool:
        """
        Check if a provider has any restrictions.

        Args:
            provider_type: The provider type

        Returns:
            True if restrictions exist, False otherwise
        """
        return provider_type in self.restrictions

    def filter_models(self, provider_type: ProviderType, models: list[str]) -> list[str]:
        """
        Filter a list of models based on restrictions.

        Args:
            provider_type: The provider type
            models: List of model names to filter

        Returns:
            Filtered list containing only allowed models
        """
        if not self.has_restrictions(provider_type):
            return models

        return [m for m in models if self.is_allowed(provider_type, m)]

    def get_restriction_summary(self) -> dict[str, any]:
        """
        Get a summary of all restrictions for logging/debugging.

        Returns:
            Dictionary with provider names and their restrictions
        """
        summary = {}
        for provider_type, allowed_set in self.restrictions.items():
            if allowed_set:
                summary[provider_type.value] = sorted(allowed_set)
            else:
                summary[provider_type.value] = "none (provider disabled)"

        return summary


# Global instance (singleton pattern)
_restriction_service: ModelRestrictionService | None = None


def get_restriction_service() -> ModelRestrictionService:
    """
    Get the global restriction service instance.

    Returns:
        The singleton ModelRestrictionService instance
    """
    global _restriction_service
    if _restriction_service is None:
        _restriction_service = ModelRestrictionService()
    return _restriction_service
