"""OpenCode Go provider implementation.

OpenCode Go (https://opencode.ai/docs/go/) is a flat-rate subscription gateway
from the OpenCode (SST) team that serves a curated set of open-source coding
models over an OpenAI-compatible API. Because every Go model is an open-weight
model, the whole catalogue is reachable through the OpenAI-compatible
``/chat/completions`` endpoint - there is no native-protocol split, so the
shared :class:`OpenAICompatibleProvider` pipeline handles it end to end.

Model metadata is sourced from models.dev (the same database OpenCode itself
uses) and pinned in ``conf/opencode_go_models.json``.
"""

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.opencode_go import OpenCodeGoModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class OpenCodeGoProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for the OpenCode Go subscription gateway.

    Exposes the curated Go catalogue (GLM, Kimi, DeepSeek, Qwen, MiniMax, MiMo)
    via the OpenAI-compatible endpoint and maps tool-category preferences to
    sensible defaults from that set.
    """

    FRIENDLY_NAME = "OpenCode Go"

    REGISTRY_CLASS = OpenCodeGoModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Fixed gateway endpoint (models.dev: provider 'opencode-go').
    DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"

    # Canonical identifiers used for category routing.
    PRIMARY_MODEL = "deepseek-v4-pro"
    FALLBACK_MODEL = "glm-5.2"
    FAST_MODEL = "deepseek-v4-flash"

    def __init__(self, api_key: str, **kwargs):
        """Initialize the OpenCode Go provider with a subscription API key."""
        kwargs.setdefault("base_url", self.DEFAULT_BASE_URL)
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    @classmethod
    def from_env(cls) -> "OpenCodeGoProvider | None":
        """Construct a provider from environment variables.

        Reads ``OPENCODE_API_KEY`` (the same variable OpenCode itself uses) and
        rejects the documented placeholder value.

        Returns:
            A configured provider instance, or ``None`` when the API key is
            missing or set to the placeholder string.
        """
        api_key = get_env("OPENCODE_API_KEY")
        if not api_key or api_key == "your_opencode_api_key_here":
            return None
        return cls(api_key=api_key)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENCODE_GO

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get OpenCode Go's preferred model for a given tool category.

        Args:
            category: The tool category requiring a model.
            allowed_models: Pre-filtered list of models allowed by restrictions.

        Returns:
            Preferred model name or ``None``.
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.FAST_RESPONSE:
            preference = [self.FAST_MODEL, self.FALLBACK_MODEL, self.PRIMARY_MODEL]
        else:  # EXTENDED_REASONING, BALANCED, or default
            preference = [self.PRIMARY_MODEL, self.FALLBACK_MODEL, self.FAST_MODEL]

        for model in preference:
            if model in allowed_models:
                return model
        return allowed_models[0]


# Load registry data at import time
OpenCodeGoProvider._ensure_registry()
