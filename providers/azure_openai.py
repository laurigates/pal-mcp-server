"""Azure OpenAI provider built on the OpenAI-compatible implementation."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from typing import Any

try:  # pragma: no cover - optional dependency
    from openai import AsyncAzureOpenAI
except ImportError:  # pragma: no cover
    AsyncAzureOpenAI = None  # type: ignore[assignment]

from utils.env import get_env, suppress_env_vars

from .openai import OpenAIModelProvider
from .openai_compatible import OpenAICompatibleProvider
from .registries.azure import AzureModelRegistry
from .shared import ModelCapabilities, ModelResponse, ProviderType, TemperatureConstraint

logger = logging.getLogger(__name__)


class AzureOpenAIProvider(OpenAICompatibleProvider):
    """Thin Azure wrapper that reuses the OpenAI-compatible request pipeline."""

    FRIENDLY_NAME = "Azure OpenAI"
    DEFAULT_API_VERSION = "2024-02-15-preview"

    MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {}

    @classmethod
    def from_env(cls) -> AzureOpenAIProvider | None:
        """Construct a provider from environment variables.

        Requires both ``AZURE_OPENAI_API_KEY`` (non-placeholder) and
        ``AZURE_OPENAI_ENDPOINT``. Also verifies the Azure model registry
        contains at least one configured deployment; otherwise returns
        ``None`` with a warning so the rest of the server keeps booting.

        Returns:
            A configured provider instance, or ``None`` when configuration
            is incomplete or no Azure models are configured.
        """
        api_key = get_env("AZURE_OPENAI_API_KEY")
        if not api_key or api_key == "your_azure_openai_key_here":
            return None
        azure_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT missing - skipping Azure OpenAI provider")
            return None
        try:
            registry = AzureModelRegistry()
            if not registry.list_models():
                logger.warning(
                    "Azure OpenAI models configuration is empty. "
                    "Populate conf/azure_models.json or set AZURE_MODELS_CONFIG_PATH."
                )
                return None
        except Exception as exc:
            logger.warning("Failed to load Azure OpenAI models: %s", exc)
            return None
        api_version = get_env("AZURE_OPENAI_API_VERSION")
        try:
            return cls(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        except Exception as exc:
            logger.warning("Failed to instantiate Azure OpenAI provider: %s", exc)
            return None

    def __init__(
        self,
        api_key: str,
        *,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployments: dict[str, object] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(api_key, base_url=azure_endpoint, **kwargs)
        if not azure_endpoint:
            azure_endpoint = get_env("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required via parameter or AZURE_OPENAI_ENDPOINT")
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.api_version = api_version or get_env("AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION)
        registry_specs = self._load_registry_entries()
        override_specs = self._normalise_deployments(deployments or {}) if deployments else {}
        self._model_specs = self._merge_specs(registry_specs, override_specs)
        if not self._model_specs:
            raise ValueError(
                "Azure OpenAI provider requires at least one configured deployment. "
                "Populate conf/azure_models.json or set AZURE_MODELS_CONFIG_PATH."
            )
        self._capabilities = self._build_capabilities_map()
        self._deployment_map = {name: spec["deployment"] for name, spec in self._model_specs.items()}
        self._deployment_alias_lookup = {
            deployment.lower(): canonical for canonical, deployment in self._deployment_map.items()
        }
        self._canonical_lookup = {name.lower(): name for name in self._model_specs.keys()}
        self._invalidate_capability_cache()

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        return dict(self._capabilities)

    def get_provider_type(self) -> ProviderType:
        return ProviderType.AZURE

    def get_capabilities(self, model_name: str) -> ModelCapabilities:  # type: ignore[override]
        lowered = model_name.lower()
        if lowered in self._deployment_alias_lookup:
            canonical = self._deployment_alias_lookup[lowered]
            return super().get_capabilities(canonical)
        canonical = self._canonical_lookup.get(lowered)
        if canonical:
            return super().get_capabilities(canonical)
        return super().get_capabilities(model_name)

    def validate_model_name(self, model_name: str) -> bool:  # type: ignore[override]
        lowered = model_name.lower()
        if lowered in self._deployment_alias_lookup or lowered in self._canonical_lookup:
            return True
        return super().validate_model_name(model_name)

    def _build_capabilities_map(self) -> dict[str, ModelCapabilities]:
        capabilities: dict[str, ModelCapabilities] = {}
        for canonical_name, spec in self._model_specs.items():
            template_capability: ModelCapabilities | None = spec.get("capability")
            overrides = spec.get("overrides", {})
            if template_capability:
                cloned = replace(template_capability)
            else:
                template = OpenAIModelProvider.MODEL_CAPABILITIES.get(canonical_name)
                if template:
                    friendly = template.friendly_name.replace("OpenAI", "Azure OpenAI", 1)
                    cloned = replace(
                        template,
                        provider=ProviderType.AZURE,
                        friendly_name=friendly,
                        aliases=list(template.aliases),
                    )
                else:
                    deployment_name = spec.get("deployment", "")
                    cloned = ModelCapabilities(
                        provider=ProviderType.AZURE,
                        model_name=canonical_name,
                        friendly_name=f"Azure OpenAI ({canonical_name})",
                        description=f"Azure deployment SQ{deployment_name}SQ for {canonical_name}".replace(
                            "SQ", chr(39)
                        ),
                        aliases=[],
                    )
            if overrides:
                overrides = dict(overrides)
                temp_override = overrides.get("temperature_constraint")
                if isinstance(temp_override, str):
                    overrides["temperature_constraint"] = TemperatureConstraint.create(temp_override)
                aliases_override = overrides.get("aliases")
                if isinstance(aliases_override, str):
                    overrides["aliases"] = [alias.strip() for alias in aliases_override.split(",") if alias.strip()]
                provider_override = overrides.get("provider")
                if provider_override:
                    overrides.pop("provider", None)
                try:
                    cloned = replace(cloned, **overrides)
                except TypeError:
                    base_data = asdict(cloned)
                    base_data.update(overrides)
                    base_data["provider"] = ProviderType.AZURE
                    temp_value = base_data.get("temperature_constraint")
                    if isinstance(temp_value, str):
                        base_data["temperature_constraint"] = TemperatureConstraint.create(temp_value)
                    cloned = ModelCapabilities(**base_data)
            if cloned.provider != ProviderType.AZURE:
                cloned.provider = ProviderType.AZURE
            capabilities[canonical_name] = cloned
        return capabilities

    def _load_registry_entries(self) -> dict[str, dict]:
        try:
            registry = AzureModelRegistry()
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to load Azure model registry: %s", exc)
            return {}
        entries: dict[str, dict] = {}
        for model_name, capability, extra in registry.iter_entries():
            deployment = extra.get("deployment")
            if not deployment:
                logger.warning("Azure model SQ%sSQ missing deployment in registry".replace("SQ", chr(39)), model_name)
                continue
            entries[model_name] = {"deployment": deployment, "capability": capability}
        return entries

    @staticmethod
    def _merge_specs(
        registry_specs: dict[str, dict],
        override_specs: dict[str, dict],
    ) -> dict[str, dict]:
        specs: dict[str, dict] = {}
        for canonical, entry in registry_specs.items():
            specs[canonical] = {
                "deployment": entry.get("deployment"),
                "capability": entry.get("capability"),
                "overrides": {},
            }
        for canonical, entry in override_specs.items():
            spec = specs.get(canonical, {"deployment": None, "capability": None, "overrides": {}})
            deployment = entry.get("deployment")
            if deployment:
                spec["deployment"] = deployment
            overrides = {k: v for k, v in entry.items() if k not in {"deployment"}}
            overrides.pop("capability", None)
            if overrides:
                spec["overrides"].update(overrides)
            specs[canonical] = spec
        return {k: v for k, v in specs.items() if v.get("deployment")}

    @staticmethod
    def _normalise_deployments(mapping: dict[str, object]) -> dict[str, dict]:
        normalised: dict[str, dict] = {}
        for canonical, spec in mapping.items():
            canonical_name = (canonical or "").strip()
            if not canonical_name:
                continue
            deployment_name: str | None = None
            overrides: dict[str, object] = {}
            if isinstance(spec, str):
                deployment_name = spec.strip()
            elif isinstance(spec, dict):
                deployment_name = spec.get("deployment") or spec.get("deployment_name")
                overrides = {k: v for k, v in spec.items() if k not in {"deployment", "deployment_name"}}
            if not deployment_name:
                continue
            normalised[canonical_name] = {"deployment": deployment_name.strip(), **overrides}
        return normalised

    @property
    def client(self):  # type: ignore[override]
        """Instantiate the Azure OpenAI client on first use."""
        if self._client is None:
            if AsyncAzureOpenAI is None:
                raise ImportError(
                    "Azure OpenAI support requires the SQopenaiSQ package. Install it with SQpip install openaiSQ.".replace(
                        "SQ", chr(39)
                    )
                )
            import httpx

            proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
            with suppress_env_vars(*proxy_env_vars):
                try:
                    timeout_config = self.timeout_config
                    http_client = httpx.AsyncClient(timeout=timeout_config, follow_redirects=True)
                    client_kwargs = {
                        "api_key": self.api_key,
                        "azure_endpoint": self.azure_endpoint,
                        "api_version": self.api_version,
                        "http_client": http_client,
                    }
                    if self.DEFAULT_HEADERS:
                        client_kwargs["default_headers"] = self.DEFAULT_HEADERS.copy()
                    self._client = AsyncAzureOpenAI(**client_kwargs)
                except Exception as exc:
                    logger.error("Failed to create Azure OpenAI client: %s", exc)
                    raise
        return self._client

    def _resolve_canonical_and_deployment(self, model_name: str) -> tuple[str, str]:
        resolved_canonical = self._resolve_model_name(model_name)
        if resolved_canonical not in self._deployment_map:
            for canonical, deployment in self._deployment_map.items():
                if deployment.lower() == resolved_canonical.lower():
                    return canonical, deployment
            raise ValueError(f"Model SQ{model_name}SQ is not configured for Azure OpenAI".replace("SQ", chr(39)))
        return resolved_canonical, self._deployment_map[resolved_canonical]

    def _build_request(
        self,
        *,
        prompt: str,
        model_name: str,
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int | None,
        images: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Translate Azure model name into deployment-keyed OpenAI-compatible payload."""
        canonical_name, deployment_name = self._resolve_canonical_and_deployment(model_name)
        request = super()._build_request(
            prompt=prompt,
            model_name=deployment_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            images=images,
            **kwargs,
        )
        request["canonical_name"] = canonical_name
        request["deployment_name"] = deployment_name
        return request

    def _parse_response(self, raw: Any, *, model_name: str, request: dict[str, Any]) -> ModelResponse:
        """Reuse OpenAI-compatible parsing then remap to the canonical Azure name."""
        raw_response = super()._parse_response(raw, model_name=model_name, request=request)
        canonical_name = request.get("canonical_name", model_name)
        deployment_name = request.get("deployment_name", request.get("model"))
        capabilities = self._capabilities.get(canonical_name)
        friendly_name = capabilities.friendly_name if capabilities else self.FRIENDLY_NAME
        return ModelResponse(
            content=raw_response.content,
            usage=raw_response.usage,
            model_name=canonical_name,
            friendly_name=friendly_name,
            provider=ProviderType.AZURE,
            metadata={**raw_response.metadata, "deployment": deployment_name},
        )

    def _parse_allowed_models(self) -> set[str] | None:  # type: ignore[override]
        explicit = get_env("AZURE_OPENAI_ALLOWED_MODELS")
        if explicit:
            models = {m.strip().lower() for m in explicit.split(",") if m.strip()}
            if models:
                logger.info("Configured allowed models for Azure OpenAI: %s", sorted(models))
                self._allowed_alias_cache = {}
                return models
        return super()._parse_allowed_models()
