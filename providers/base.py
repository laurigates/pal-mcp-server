"""Base interfaces and common behaviour for model providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for model backends."""

    MODEL_CAPABILITIES: dict[str, Any] = {}
    MAX_RETRIES: ClassVar[int] = 4
    RETRY_DELAYS: ClassVar[list[float]] = [1.0, 3.0, 5.0, 8.0]

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.config = kwargs
        self._sorted_capabilities_cache: list[tuple[str, ModelCapabilities]] | None = None

    @abstractmethod
    def get_provider_type(self) -> ProviderType:
        """Return the concrete provider identity."""

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Resolve capability metadata for a model name."""
        resolved_model_name = self._resolve_model_name(model_name)
        capabilities = self._lookup_capabilities(resolved_model_name, model_name)
        if capabilities is None:
            self._raise_unsupported_model(model_name)
        self._ensure_model_allowed(capabilities, resolved_model_name, model_name)
        return self._finalise_capabilities(capabilities, resolved_model_name, model_name)

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        """Return statically declared capabilities when available."""
        model_map = getattr(self, "MODEL_CAPABILITIES", None)
        if isinstance(model_map, dict) and model_map:
            return {k: v for k, v in model_map.items() if isinstance(v, ModelCapabilities)}
        return {}

    def get_capabilities_by_rank(self) -> list[tuple[str, ModelCapabilities]]:
        """Return model capabilities sorted by effective capability rank."""
        if self._sorted_capabilities_cache is not None:
            return list(self._sorted_capabilities_cache)
        model_configs = self.get_all_model_capabilities()
        if not model_configs:
            self._sorted_capabilities_cache = []
            return []
        items = list(model_configs.items())
        items.sort(key=lambda item: (-item[1].get_effective_capability_rank(), item[0]))
        self._sorted_capabilities_cache = items
        return list(items)

    def _invalidate_capability_cache(self) -> None:
        """Clear cached sorted capability data."""
        self._sorted_capabilities_cache = None

    def list_models(
        self,
        *,
        respect_restrictions: bool = True,
        include_aliases: bool = True,
        lowercase: bool = False,
        unique: bool = False,
    ) -> list[str]:
        """Return formatted model names supported by this provider."""
        model_configs = self.get_all_model_capabilities()
        if not model_configs:
            return []
        restriction_service = None
        if respect_restrictions:
            from utils.model_restrictions import get_restriction_service

            restriction_service = get_restriction_service()
        if restriction_service:
            allowed_configs = {}
            for model_name, config in model_configs.items():
                if restriction_service.is_allowed(self.get_provider_type(), model_name):
                    allowed_configs[model_name] = config
            model_configs = allowed_configs
        if not model_configs:
            return []
        return ModelCapabilities.collect_model_names(
            model_configs,
            include_aliases=include_aliases,
            lowercase=lowercase,
            unique=unique,
        )

    async def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Template method orchestrating build/call/parse with retries."""
        self.validate_parameters(model_name, temperature)
        request = self._build_request(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        attempt_counter = {"value": 0}

        async def _execute() -> Any:
            attempt_counter["value"] += 1
            return await asyncio.to_thread(self._call_api, request)

        log_prefix = self._retry_log_prefix(request, model_name)
        try:
            raw = await self._run_with_retries_async(
                operation=_execute,
                max_attempts=self.MAX_RETRIES,
                delays=self.RETRY_DELAYS,
                log_prefix=log_prefix,
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            raise self._wrap_api_failure(exc, request=request, model_name=model_name, attempts=attempts) from exc
        return self._parse_response(raw, model_name=model_name, request=request)

    @abstractmethod
    def _build_request(
        self,
        *,
        prompt: str,
        model_name: str,
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Translate user-facing arguments into a provider-shaped request dict."""

    @abstractmethod
    def _call_api(self, request: dict[str, Any]) -> Any:
        """Invoke the underlying SDK using request and return the raw response."""

    @abstractmethod
    def _parse_response(
        self,
        raw: Any,
        *,
        model_name: str,
        request: dict[str, Any],
    ) -> ModelResponse:
        """Convert the SDK response into a ModelResponse."""

    def _retry_log_prefix(self, request: dict[str, Any], model_name: str) -> str:
        """Compute the log prefix used while retrying."""
        resolved = request.get("model") or model_name
        return f"{self.__class__.__name__} ({resolved})"

    def _wrap_api_failure(
        self,
        exc: Exception,
        *,
        request: dict[str, Any],
        model_name: str,
        attempts: int,
    ) -> Exception:
        """Wrap a terminal SDK failure into a user-facing exception."""
        provider_name = getattr(self, "FRIENDLY_NAME", None) or self.__class__.__name__
        resolved = request.get("model") or model_name
        suffix = "s" if attempts != 1 else ""
        return RuntimeError(f"{provider_name} API error for model {resolved} after {attempts} attempt{suffix}: {exc}")

    def count_tokens(self, text: str, model_name: str) -> int:
        """Estimate token usage for a piece of text."""
        resolved_model = self._resolve_model_name(model_name)
        if not text:
            return 0
        estimated = max(1, len(text) // 4)
        logger.debug("Estimating %s tokens for model %s via character heuristic", estimated, resolved_model)
        return estimated

    def close(self) -> None:
        """Clean up any resources held by the provider."""
        return

    def _is_error_retryable(self, error: Exception) -> bool:
        """Return True when an error warrants another attempt."""
        error_str = str(error).lower()
        if "429" in error_str or "rate limit" in error_str:
            return False
        retryable_indicators = [
            "timeout",
            "connection",
            "temporary",
            "unavailable",
            "retry",
            "reset",
            "refused",
            "broken pipe",
            "tls",
            "handshake",
            "network",
            "500",
            "502",
            "503",
            "504",
        ]
        return any(indicator in error_str for indicator in retryable_indicators)

    async def _run_with_retries_async(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        max_attempts: int,
        delays: list[float] | None = None,
        log_prefix: str = "",
    ):
        """Async retry helper for awaitable operations."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        attempts = max_attempts
        delays = delays or []
        last_exc: Exception | None = None
        for attempt_index in range(attempts):
            try:
                return await operation()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempt_number = attempt_index + 1
                retryable = self._is_error_retryable(exc)
                if not retryable or attempt_number >= attempts:
                    raise
                delay_idx = min(attempt_index, len(delays) - 1) if delays else -1
                delay = delays[delay_idx] if delay_idx >= 0 else 0.0
                if delay > 0:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying in %ss...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        "%s retryable error (attempt %s/%s): %s. Retrying...",
                        log_prefix or self.__class__.__name__,
                        attempt_number,
                        attempts,
                        exc,
                    )
        raise last_exc if last_exc else RuntimeError("Retry loop exited without result")

    def validate_model_name(self, model_name: str) -> bool:
        """Return True when the model resolves to an allowed capability."""
        try:
            self.get_capabilities(model_name)
        except ValueError:
            return False
        return True

    def validate_parameters(self, model_name: str, temperature: float, **kwargs) -> None:
        """Validate model parameters against capabilities."""
        capabilities = self.get_capabilities(model_name)
        if not capabilities.temperature_constraint.validate(temperature):
            constraint_desc = capabilities.temperature_constraint.get_description()
            raise ValueError(f"Temperature {temperature} is invalid for model {model_name}. {constraint_desc}")

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get the preferred model from this provider for a given category."""
        return None

    def get_model_registry(self) -> dict[str, Any] | None:
        """Return the model registry backing this provider, if any."""
        return None

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        """Return ModelCapabilities for the canonical model name."""
        return self.get_all_model_capabilities().get(canonical_name)

    def _ensure_model_allowed(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> None:
        """Raise ValueError if the model violates restriction policy."""
        try:
            from utils.model_restrictions import get_restriction_service
        except Exception:
            return
        restriction_service = get_restriction_service()
        if not restriction_service:
            return
        if restriction_service.is_allowed(self.get_provider_type(), canonical_name, requested_name):
            return
        raise ValueError(
            f"{self.get_provider_type().value} model SQ{canonical_name}SQ is not allowed by restriction policy.".replace(
                "SQ", chr(39)
            )
        )

    def _finalise_capabilities(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> ModelCapabilities:
        """Allow subclasses to adjust capability metadata before returning."""
        return capabilities

    def _raise_unsupported_model(self, model_name: str) -> None:
        """Raise the canonical unsupported-model error."""
        raise ValueError(
            f"Unsupported model SQ{model_name}SQ for provider {self.get_provider_type().value}.".replace("SQ", chr(39))
        )

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model shorthand to full name."""
        model_configs = self.get_all_model_capabilities()
        if model_name in model_configs:
            return model_name
        model_name_lower = model_name.lower()
        for base_model in model_configs:
            if base_model.lower() == model_name_lower:
                return base_model
        alias_map = ModelCapabilities.collect_aliases(model_configs)
        for base_model, aliases in alias_map.items():
            if any(alias.lower() == model_name_lower for alias in aliases):
                return base_model
        return model_name
