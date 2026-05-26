"""Custom API provider implementation."""

import logging

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.custom import CustomEndpointModelRegistry
from .registries.openrouter import OpenRouterModelRegistry
from .shared import ModelCapabilities, ProviderType


class CustomProvider(OpenAICompatibleProvider):
    """Adapter for self-hosted or local OpenAI-compatible endpoints.

    Role
        Provide a uniform bridge between the MCP server and user-managed
        OpenAI-compatible services (Ollama, vLLM, LM Studio, bespoke gateways).
        By subclassing :class:`OpenAICompatibleProvider` it inherits request and
        token handling, while the custom registry exposes locally defined model
        metadata.

    Notable behaviour
        * Uses :class:`OpenRouterModelRegistry` to load model definitions and
          aliases so custom deployments share the same metadata pipeline as
          OpenRouter itself.
        * Normalises version-tagged model names (``model:latest``) and applies
          restriction policies just like cloud providers, ensuring consistent
          behaviour across environments.
    """

    FRIENDLY_NAME = "Custom API"

    # Model registry for managing configurations and aliases
    _registry: CustomEndpointModelRegistry | None = None

    @classmethod
    def from_env(cls) -> "CustomProvider | None":
        """Construct a provider from environment variables.

        Requires ``CUSTOM_API_URL`` to point at the OpenAI-compatible
        endpoint. ``CUSTOM_API_KEY`` is optional - some endpoints (Ollama)
        accept unauthenticated requests, so an empty key is honoured.

        Returns:
            A configured provider instance, or ``None`` when
            ``CUSTOM_API_URL`` is not set.
        """
        base_url = get_env("CUSTOM_API_URL", "") or ""
        if not base_url:
            return None
        api_key = get_env("CUSTOM_API_KEY", "") or ""
        try:
            return cls(api_key=api_key, base_url=base_url)
        except Exception as exc:
            logging.warning("Failed to instantiate Custom provider: %s", exc)
            return None

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs):
        """Initialize Custom provider for local/self-hosted models.

        This provider supports any OpenAI-compatible API endpoint including:
        - Ollama (typically no API key required)
        - vLLM (may require API key)
        - LM Studio (may require API key)
        - Text Generation WebUI (may require API key)
        - Enterprise/self-hosted APIs (typically require API key)

        Args:
            api_key: API key for the custom endpoint. Can be empty string for
                    providers that don't require authentication (like Ollama).
                    Falls back to CUSTOM_API_KEY environment variable if not provided.
            base_url: Base URL for the custom API endpoint (e.g., 'http://localhost:11434/v1').
                     Falls back to CUSTOM_API_URL environment variable if not provided.
            **kwargs: Additional configuration passed to parent OpenAI-compatible provider

        Raises:
            ValueError: If no base_url is provided via parameter or environment variable
        """
        # Fall back to environment variables only if not provided
        if not base_url:
            base_url = get_env("CUSTOM_API_URL", "") or ""
        if not api_key:
            api_key = get_env("CUSTOM_API_KEY", "") or ""

        if not base_url:
            raise ValueError(
                "Custom API URL must be provided via base_url parameter or CUSTOM_API_URL environment variable"
            )

        # For Ollama and other providers that don't require authentication,
        # set a dummy API key to avoid OpenAI client header issues
        if not api_key:
            api_key = "dummy-key-for-unauthenticated-endpoint"
            logging.debug("Using dummy API key for unauthenticated custom endpoint")

        logging.info(f"Initializing Custom provider with endpoint: {base_url}")

        self._alias_cache: dict[str, str] = {}

        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if CustomProvider._registry is None:
            CustomProvider._registry = CustomEndpointModelRegistry()
            # Log loaded models and aliases only on first load
            registry = CustomProvider._registry
            models = registry.list_models()
            aliases = registry.list_aliases()
            logging.info(f"Custom provider loaded {len(models)} models with {len(aliases)} aliases")

    # ------------------------------------------------------------------
    # Registry accessor (lazy materialisation)
    # ------------------------------------------------------------------
    @classmethod
    def _get_registry(cls) -> CustomEndpointModelRegistry:
        """Return the shared model registry, materialising it on first access.

        The class-level ``_registry`` attribute is typed as ``Optional`` to allow
        deferred construction, but every code path that needs it must see a
        non-Optional reference. This accessor performs the lazy initialisation
        once and guarantees a concrete registry to callers, eliminating the
        ``Optional[...]`` from the call sites.
        """
        if cls._registry is None:
            cls._registry = CustomEndpointModelRegistry()
        return cls._registry

    # ------------------------------------------------------------------
    # Capability surface
    # ------------------------------------------------------------------
    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        """Return capabilities for models explicitly marked as custom."""

        builtin = super()._lookup_capabilities(canonical_name, requested_name)
        if builtin is not None:
            return builtin

        registry_entry = self._get_registry().resolve(canonical_name)
        if registry_entry:
            registry_entry.provider = ProviderType.CUSTOM
            return registry_entry

        logging.debug(
            "Custom provider cannot resolve model '%s'; ensure it is declared in custom_models.json",
            canonical_name,
        )
        return None

    def get_provider_type(self) -> ProviderType:
        """Identify this provider for restriction and logging logic."""

        return ProviderType.CUSTOM

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve registry aliases and strip version tags for local models."""

        cache_key = model_name.lower()
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        registry = self._get_registry()
        config = registry.resolve(model_name)
        if config:
            if config.model_name != model_name:
                logging.debug("Resolved model alias '%s' to '%s'", model_name, config.model_name)
            resolved = config.model_name
            self._alias_cache[cache_key] = resolved
            self._alias_cache.setdefault(resolved.lower(), resolved)
            return resolved

        if ":" in model_name:
            base_model = model_name.split(":")[0]
            logging.debug(f"Stripped version tag from '{model_name}' -> '{base_model}'")

            base_config = registry.resolve(base_model)
            if base_config:
                logging.debug("Resolved base model '%s' to '%s'", base_model, base_config.model_name)
                resolved = base_config.model_name
                self._alias_cache[cache_key] = resolved
                self._alias_cache.setdefault(resolved.lower(), resolved)
                return resolved
            self._alias_cache[cache_key] = base_model
            return base_model

        logging.debug(f"Model '{model_name}' not found in registry, using as-is")
        # Attempt to resolve via OpenRouter registry so aliases still map cleanly
        openrouter_registry = OpenRouterModelRegistry()
        openrouter_config = openrouter_registry.resolve(model_name)
        if openrouter_config:
            resolved = openrouter_config.model_name
            self._alias_cache[cache_key] = resolved
            self._alias_cache.setdefault(resolved.lower(), resolved)
            return resolved

        self._alias_cache[cache_key] = model_name
        return model_name

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        """Expose registry capabilities for models marked as custom."""

        registry = self._get_registry()

        capabilities = {}
        for model in registry.list_models():
            config = registry.resolve(model)
            if config:
                capabilities[model] = config
        return capabilities
