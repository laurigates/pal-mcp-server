"""X.AI (GROK) model provider implementation."""

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory


from .openai_compatible import OpenAICompatibleProvider
from .registries.xai import XAIModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class XAIModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for X.AI's GROK models exposed over an OpenAI-style API.

    Publishes capability metadata for the officially supported deployments and
    maps tool-category preferences to the appropriate GROK model.
    """

    PROVIDER_TYPE = ProviderType.XAI
    FRIENDLY_NAME = "X.AI"
    DISPLAY_NAME = "X.AI (Grok)"
    HELP_SUMMARY = "X.AI GROK models"
    API_KEY_ENV = "XAI_API_KEY"
    API_KEY_PLACEHOLDER = "your_xai_api_key_here"
    OPTIONAL_ENV = ("XAI_MODELS_CONFIG_PATH",)

    REGISTRY_CLASS = XAIModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Canonical model identifiers used for category routing.
    # These must name models that are still live in conf/xai_models.json - when
    # the registry drops or supersedes a model, move these in the same change.
    PRIMARY_MODEL = "grok-4.6"
    FALLBACK_MODEL = "grok-4.5"

    def __init__(self, api_key: str, **kwargs):
        """Initialize X.AI provider with API key."""
        # Set X.AI base URL
        kwargs.setdefault("base_url", "https://api.x.ai/v1")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get XAI's preferred model for a given category from allowed models.

        Args:
            category: The tool category requiring a model
            allowed_models: Pre-filtered list of models allowed by restrictions

        Returns:
            Preferred model name or None
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.EXTENDED_REASONING:
            # Prefer Grok 4.6 for advanced tasks
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        elif category == ToolModelCategory.FAST_RESPONSE:
            # Prefer Grok 4.6 for speed as well (latest flagship SKU).
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]

        else:  # BALANCED or default
            # Prefer Grok 4.6 for balanced use.
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            if self.FALLBACK_MODEL in allowed_models:
                return self.FALLBACK_MODEL
            return allowed_models[0]


# Load registry data at import time
XAIModelProvider._ensure_registry()
