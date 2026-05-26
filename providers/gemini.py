"""Gemini model provider implementation."""

import base64
import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from google import genai
from google.genai import types

from utils.env import get_env
from utils.image_utils import validate_image

from .base import ModelProvider
from .registries.gemini import GeminiModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class GeminiModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Gemini integration built on the official Google SDK."""

    REGISTRY_CLASS = GeminiModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}
    FRIENDLY_NAME = "Gemini"

    THINKING_BUDGETS = {
        "minimal": 0.005,
        "low": 0.08,
        "medium": 0.33,
        "high": 0.67,
        "max": 1.0,
    }

    def __init__(self, api_key: str, **kwargs):
        """Initialize Gemini provider with API key and optional base URL."""
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._client = None
        self._token_counters = {}
        self._base_url = kwargs.get("base_url", None)
        self._timeout_override = self._resolve_http_timeout()
        self._invalidate_capability_cache()

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            http_options_kwargs: dict[str, object] = {}
            if self._base_url:
                http_options_kwargs["base_url"] = self._base_url
            if self._timeout_override is not None:
                http_options_kwargs["timeout"] = self._timeout_override
            if http_options_kwargs:
                http_options = types.HttpOptions(**http_options_kwargs)
                logger.debug(
                    "Initializing Gemini client with options: base_url=%s timeout=%s",
                    http_options_kwargs.get("base_url"),
                    http_options_kwargs.get("timeout"),
                )
                self._client = genai.Client(api_key=self.api_key, http_options=http_options)
            else:
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _resolve_http_timeout(self) -> float | None:
        """Compute timeout override from shared custom timeout environment variables."""
        timeouts: list[float] = []
        for env_var in [
            "CUSTOM_CONNECT_TIMEOUT",
            "CUSTOM_READ_TIMEOUT",
            "CUSTOM_WRITE_TIMEOUT",
            "CUSTOM_POOL_TIMEOUT",
        ]:
            raw_value = get_env(env_var)
            if raw_value:
                try:
                    timeouts.append(float(raw_value))
                except (TypeError, ValueError):
                    logger.warning("Invalid %s value SQ%sSQ; ignoring.".replace("SQ", chr(39)), env_var, raw_value)
        if timeouts:
            resolved = max(timeouts)
            logger.debug("Using custom Gemini HTTP timeout: %ss", resolved)
            return resolved
        return None

    def get_provider_type(self) -> ProviderType:
        return ProviderType.GOOGLE

    def _build_request(
        self,
        *,
        prompt: str,
        model_name: str,
        system_prompt: str | None,
        temperature: float,
        max_output_tokens: int | None,
        thinking_mode: str = "medium",
        images: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Assemble the Gemini contents/config payload."""
        capabilities = self.get_capabilities(model_name)
        capability_map = self.get_all_model_capabilities()
        resolved_model_name = self._resolve_model_name(model_name)

        parts: list[dict[str, Any]] = []
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        parts.append({"text": full_prompt})

        if images and capabilities.supports_images:
            for image_path in images:
                try:
                    image_part = self._process_image(image_path)
                    if image_part:
                        parts.append(image_part)
                except Exception as e:
                    logger.warning(f"Failed to process image {image_path}: {e}")
                    continue
        elif images and not capabilities.supports_images:
            logger.warning(f"Model {resolved_model_name} does not support images, ignoring {len(images)} image(s)")

        contents = [{"parts": parts}]

        effective_thinking_mode = thinking_mode
        if resolved_model_name == "gemini-3-pro-preview" and thinking_mode == "medium":
            logger.debug(
                "Overriding thinking mode SQmediumSQ with SQhighSQ for %s due to launch limitation".replace(
                    "SQ", chr(39)
                ),
                resolved_model_name,
            )
            effective_thinking_mode = "high"

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=1,
        )
        if max_output_tokens:
            generation_config.max_output_tokens = max_output_tokens
        if capabilities.supports_extended_thinking and effective_thinking_mode in self.THINKING_BUDGETS:
            model_config = capability_map.get(resolved_model_name)
            if model_config and model_config.max_thinking_tokens > 0:
                max_thinking_tokens = model_config.max_thinking_tokens
                actual_thinking_budget = int(max_thinking_tokens * self.THINKING_BUDGETS[effective_thinking_mode])
                generation_config.thinking_config = types.ThinkingConfig(thinking_budget=actual_thinking_budget)

        return {
            "model": resolved_model_name,
            "contents": contents,
            "config": generation_config,
            "effective_thinking_mode": effective_thinking_mode,
            "supports_extended_thinking": capabilities.supports_extended_thinking,
        }

    def _call_api(self, request: dict[str, Any]) -> Any:
        """Invoke the Gemini SDK with the prepared request."""
        return self.client.models.generate_content(
            model=request["model"],
            contents=request["contents"],
            config=request["config"],
        )

    def _retry_log_prefix(self, request: dict[str, Any], model_name: str) -> str:
        resolved = request.get("model") or model_name
        return f"Gemini API ({resolved})"

    def _parse_response(self, raw: Any, *, model_name: str, request: dict[str, Any]) -> ModelResponse:
        """Convert a Gemini SDK response into ModelResponse."""
        resolved_model_name = request["model"]
        effective_thinking_mode = request.get("effective_thinking_mode")
        supports_extended_thinking = request.get("supports_extended_thinking", False)
        usage = self._extract_usage(raw)

        finish_reason_str = "UNKNOWN"
        is_blocked_by_safety = False
        safety_feedback_details = None

        if raw.candidates:
            candidate = raw.candidates[0]
            try:
                finish_reason_enum = candidate.finish_reason
                if finish_reason_enum:
                    try:
                        finish_reason_str = finish_reason_enum.name
                    except AttributeError:
                        finish_reason_str = str(finish_reason_enum)
                else:
                    finish_reason_str = "STOP"
            except AttributeError:
                finish_reason_str = "STOP"

            if not raw.text:
                try:
                    safety_ratings = candidate.safety_ratings
                    if safety_ratings:
                        for rating in safety_ratings:
                            try:
                                if rating.blocked:
                                    is_blocked_by_safety = True
                                    category_name = "UNKNOWN"
                                    probability_name = "UNKNOWN"
                                    try:
                                        category_name = rating.category.name
                                    except (AttributeError, TypeError):
                                        pass
                                    try:
                                        probability_name = rating.probability.name
                                    except (AttributeError, TypeError):
                                        pass
                                    safety_feedback_details = (
                                        f"Category: {category_name}, Probability: {probability_name}"
                                    )
                                    break
                            except (AttributeError, TypeError):
                                continue
                except (AttributeError, TypeError):
                    pass

        elif raw.candidates is not None and len(raw.candidates) == 0:
            is_blocked_by_safety = True
            finish_reason_str = "SAFETY"
            safety_feedback_details = "Prompt blocked, reason unavailable"
            try:
                prompt_feedback = raw.prompt_feedback
                if prompt_feedback and prompt_feedback.block_reason:
                    try:
                        block_reason_name = prompt_feedback.block_reason.name
                    except AttributeError:
                        block_reason_name = str(prompt_feedback.block_reason)
                    safety_feedback_details = f"Prompt blocked, reason: {block_reason_name}"
            except (AttributeError, TypeError):
                pass

        return ModelResponse(
            content=raw.text,
            usage=usage,
            model_name=resolved_model_name,
            friendly_name="Gemini",
            provider=ProviderType.GOOGLE,
            metadata={
                "thinking_mode": effective_thinking_mode if supports_extended_thinking else None,
                "finish_reason": finish_reason_str,
                "is_blocked_by_safety": is_blocked_by_safety,
                "safety_feedback": safety_feedback_details,
            },
        )

    def _wrap_api_failure(
        self,
        exc: Exception,
        *,
        request: dict[str, Any],
        model_name: str,
        attempts: int,
    ) -> Exception:
        resolved = request.get("model") or model_name
        suffix = "s" if attempts != 1 else ""
        return RuntimeError(f"Gemini API error for model {resolved} after {attempts} attempt{suffix}: {exc}")

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        usage: dict[str, int] = {}
        try:
            metadata = response.usage_metadata
            if metadata:
                input_tokens = None
                output_tokens = None
                try:
                    value = metadata.prompt_token_count
                    if value is not None:
                        input_tokens = value
                        usage["input_tokens"] = value
                except (AttributeError, TypeError):
                    pass
                try:
                    value = metadata.candidates_token_count
                    if value is not None:
                        output_tokens = value
                        usage["output_tokens"] = value
                except (AttributeError, TypeError):
                    pass
                if input_tokens is not None and output_tokens is not None:
                    usage["total_tokens"] = input_tokens + output_tokens
        except (AttributeError, TypeError):
            pass
        return usage

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes."""
        error_str = str(error).lower()
        if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            non_retryable_indicators = [
                "quota exceeded",
                "resource exhausted",
                "context length",
                "token limit",
                "request too large",
                "invalid request",
                "quota_exceeded",
                "resource_exhausted",
            ]
            try:
                error_details = None
                try:
                    error_details = error.details
                except AttributeError:
                    try:
                        error_details = error.reason
                    except AttributeError:
                        pass
                if error_details:
                    error_details_str = str(error_details).lower()
                    if any(indicator in error_details_str for indicator in non_retryable_indicators):
                        logger.debug(f"Non-retryable Gemini error: {error_details}")
                        return False
            except Exception:
                pass
            if any(indicator in error_str for indicator in non_retryable_indicators):
                logger.debug(f"Non-retryable Gemini error based on message: {error_str[:200]}...")
                return False
            logger.debug(f"Retryable Gemini rate limiting error: {error_str[:100]}...")
            return True

        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "retry",
            "internal error",
            "408",
            "500",
            "502",
            "503",
            "504",
            "ssl",
            "handshake",
        ]
        return any(indicator in error_str for indicator in retryable_indicators)

    def _process_image(self, image_path: str) -> dict | None:
        """Process an image for Gemini API."""
        try:
            image_bytes, mime_type = validate_image(image_path)
            if image_path.startswith("data:"):
                _, data = image_path.split(",", 1)
                return {"inline_data": {"mime_type": mime_type, "data": data}}
            else:
                image_data = base64.b64encode(image_bytes).decode()
                return {"inline_data": {"mime_type": mime_type, "data": image_data}}
        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> str | None:
        """Get Gemini preferred model for a given category from allowed models."""
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None
        capability_map = self.get_all_model_capabilities()

        def find_best(candidates: list[str]) -> str | None:
            return sorted(candidates, reverse=True)[0] if candidates else None

        if category == ToolModelCategory.EXTENDED_REASONING:
            pro_thinking = [
                m
                for m in allowed_models
                if "pro" in m and m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if pro_thinking:
                return find_best(pro_thinking)
            any_thinking = [
                m for m in allowed_models if m in capability_map and capability_map[m].supports_extended_thinking
            ]
            if any_thinking:
                return find_best(any_thinking)
            pro_models = [m for m in allowed_models if "pro" in m]
            if pro_models:
                return find_best(pro_models)
        elif category == ToolModelCategory.FAST_RESPONSE:
            flash_models = [m for m in allowed_models if "flash" in m]
            if flash_models:
                return find_best(flash_models)
        flash_models = [m for m in allowed_models if "flash" in m]
        if flash_models:
            return find_best(flash_models)
        pro_models = [m for m in allowed_models if "pro" in m]
        if pro_models:
            return find_best(pro_models)
        return find_best(allowed_models)


GeminiModelProvider._ensure_registry()
