"""Registry loader for OpenCode Go model capabilities."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class OpenCodeGoModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/opencode_go_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="OPENCODE_GO_MODELS_CONFIG_PATH",
            default_filename="opencode_go_models.json",
            provider=ProviderType.OPENCODE_GO,
            friendly_prefix="OpenCode Go ({model})",
            config_path=config_path,
        )
