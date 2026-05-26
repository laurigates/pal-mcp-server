"""Base class for OpenAI-compatible API providers."""

import copy
import ipaddress
import logging
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from utils.env import get_env, suppress_env_vars
from utils.image_utils import validate_image

from .base import ModelProvider
from .shared import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
)


class OpenAICompatibleProvider(ModelProvider):
    """Shared implementation for OpenAI API lookalikes."""

    DEFAULT_HEADERS: dict[str, str] = {}
    FRIENDLY_NAME = "OpenAI Compatible"

    def __init__(self, api_key: str, base_url: str | None = None, **kwargs):
        """Initialize the provider with API key and optional base URL."""
        self._allowed_alias_cache: dict[str, str] = {}
        super().__init__(api_key, **kwargs)
        self._client = None
        self.base_url = base_url
        self.organization = kwargs.get("organization")
        self.allowed_models = self._parse_allowed_models()
        self.timeout_config = self._configure_timeouts(**kwargs)
        if self.base_url:
            self._validate_base_url()
        if self.base_url and not self._is_localhost_url() and not api_key:
            logging.warning(
                f"Using external URL SQ{self.base_url}SQ without API key. ".replace("SQ", chr(39))
                + "This may be insecure. Consider setting an API key for authentication."
            )

    def _ensure_model_allowed(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> None:
        """Respect provider-specific allowlists before default restriction checks."""
        super()._ensure_model_allowed(capabilities, canonical_name, requested_name)
        if self.allowed_models is not None:
            requested = requested_name.lower()
            canonical = canonical_name.lower()
            if requested not in self.allowed_models and canonical not in self.allowed_models:
                allowed = False
                for allowed_entry in list(self.allowed_models):
                    normalized_resolved = self._allowed_alias_cache.get(allowed_entry)
                    if normalized_resolved is None:
                        try:
                            resolved_name = self._resolve_model_name(allowed_entry)
                        except Exception:
                            continue
                        if not resolved_name:
                            continue
                        normalized_resolved = resolved_name.lower()
                        self._allowed_alias_cache[allowed_entry] = normalized_resolved
                    if normalized_resolved == canonical:
                        allowed = True
                        self._allowed_alias_cache[canonical] = canonical
                        self.allowed_models.add(canonical)
                        break
                if not allowed:
                    raise ValueError(
                        f"Model SQ{requested_name}SQ is not allowed by restriction policy. Allowed models: {sorted(self.allowed_models)}".replace(
                            "SQ", chr(39)
                        )
                    )

    def _parse_allowed_models(self) -> set[str] | None:
        """Parse allowed models from environment variable."""
        provider_type = self.get_provider_type().value.upper()
        env_var = f"{provider_type}_ALLOWED_MODELS"
        models_str = get_env(env_var, "") or ""
        if models_str:
            models = {m.strip().lower() for m in models_str.split(",") if m.strip()}
            if models:
                logging.info(f"Configured allowed models for {self.FRIENDLY_NAME}: {sorted(models)}")
                self._allowed_alias_cache = {}
                return models
        if self.get_provider_type() not in [ProviderType.GOOGLE, ProviderType.OPENAI]:
            logging.info(
                f"Model allow-list not configured for {self.FRIENDLY_NAME} - all models permitted. "
                f"To restrict access, set {env_var} with comma-separated model names."
            )
        return None

    def _configure_timeouts(self, **kwargs):
        """Configure timeout settings."""
        import httpx

        default_connect = 30.0
        default_read = 600.0
        default_write = 600.0
        default_pool = 600.0
        if self.base_url and self._is_localhost_url():
            default_connect = 60.0
            default_read = 1800.0
            default_write = 1800.0
            default_pool = 1800.0
            logging.info(f"Using extended timeouts for local endpoint: {self.base_url}")
        elif self.base_url:
            default_connect = 45.0
            default_read = 900.0
            default_write = 900.0
            default_pool = 900.0
            logging.info(f"Using extended timeouts for custom endpoint: {self.base_url}")

        connect_timeout = kwargs.get("connect_timeout")
        if connect_timeout is None:
            connect_timeout_raw = get_env("CUSTOM_CONNECT_TIMEOUT")
            connect_timeout = float(connect_timeout_raw) if connect_timeout_raw is not None else float(default_connect)
        read_timeout = kwargs.get("read_timeout")
        if read_timeout is None:
            read_timeout_raw = get_env("CUSTOM_READ_TIMEOUT")
            read_timeout = float(read_timeout_raw) if read_timeout_raw is not None else float(default_read)
        write_timeout = kwargs.get("write_timeout")
        if write_timeout is None:
            write_timeout_raw = get_env("CUSTOM_WRITE_TIMEOUT")
            write_timeout = float(write_timeout_raw) if write_timeout_raw is not None else float(default_write)
        pool_timeout = kwargs.get("pool_timeout")
        if pool_timeout is None:
            pool_timeout_raw = get_env("CUSTOM_POOL_TIMEOUT")
            pool_timeout = float(pool_timeout_raw) if pool_timeout_raw is not None else float(default_pool)

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout)
        logging.debug(
            f"Configured timeouts - Connect: {connect_timeout}s, Read: {read_timeout}s, "
            f"Write: {write_timeout}s, Pool: {pool_timeout}s"
        )
        return timeout

    def _is_localhost_url(self) -> bool:
        """Check if the base URL points to localhost or local network."""
        if not self.base_url:
            return False
        try:
            parsed = urlparse(self.base_url)
            hostname = parsed.hostname
            if hostname in ["localhost", "127.0.0.1", "::1"]:
                return True
            if hostname:
                try:
                    ip = ipaddress.ip_address(hostname)
                    return ip.is_private or ip.is_loopback
                except ValueError:
                    pass
            return False
        except Exception:
            return False

    def _validate_base_url(self) -> None:
        """Validate base URL for security (SSRF protection)."""
        if not self.base_url:
            return
        try:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")
            if not parsed.hostname:
                raise ValueError("URL must include a hostname")
            port = parsed.port
            if port is not None and (port < 1 or port > 65535):
                raise ValueError(f"Invalid port number: {port}. Must be between 1 and 65535.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid base URL SQ{self.base_url}SQ: {str(e)}".replace("SQ", chr(39)))

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            import httpx

            proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
            with suppress_env_vars(*proxy_env_vars):
                try:
                    timeout_config = (
                        self.timeout_config
                        if hasattr(self, "timeout_config") and self.timeout_config
                        else httpx.Timeout(30.0)
                    )
                    test_transport = getattr(self, "_test_transport", None)
                    if test_transport is not None:
                        http_client = httpx.Client(
                            transport=test_transport,
                            timeout=timeout_config,
                            follow_redirects=True,
                        )
                    else:
                        http_client = httpx.Client(
                            timeout=timeout_config,
                            follow_redirects=True,
                        )
                    client_kwargs: dict[str, Any] = {
                        "api_key": self.api_key,
                        "http_client": http_client,
                    }
                    if self.base_url:
                        client_kwargs["base_url"] = self.base_url
                    if self.organization:
                        client_kwargs["organization"] = self.organization
                    if self.DEFAULT_HEADERS:
                        client_kwargs["default_headers"] = self.DEFAULT_HEADERS.copy()
                    self._client = OpenAI(**client_kwargs)
                except Exception as e:
                    logging.warning("Failed to create client: %s", e)
                    try:
                        minimal_kwargs: dict[str, Any] = {"api_key": self.api_key}
                        if self.base_url:
                            minimal_kwargs["base_url"] = self.base_url
                        self._client = OpenAI(**minimal_kwargs)
                    except Exception as fallback_error:
                        logging.error("Minimal OpenAI client creation failed: %s", fallback_error)
                        raise
        return self._client

    def _sanitize_for_logging(self, params: dict) -> dict:
        """Sanitize sensitive data from parameters before logging."""
        sanitized = copy.deepcopy(params)
        if "input" in sanitized:
            for msg in sanitized.get("input", []):
                if isinstance(msg, dict) and "content" in msg:
                    for content_item in msg.get("content", []):
                        if isinstance(content_item, dict) and "text" in content_item:
                            text = content_item["text"]
                            if len(text) > 100:
                                content_item["text"] = text[:100] + "... [truncated]"
        sanitized.pop("api_key", None)
        sanitized.pop("authorization", None)
        return sanitized

    def _safe_extract_output_text(self, response) -> str:
        """Safely extract output_text from o3-pro response with validation."""
        if not hasattr(response, "output_text"):
            raise ValueError(f"o3-pro response missing output_text field. Response type: {type(response).__name__}")
        content = response.output_text
        if content is None:
            raise ValueError("o3-pro returned None for output_text")
        if not isinstance(content, str):
            raise ValueError(f"o3-pro output_text is not a string. Got type: {type(content).__name__}")
        return content

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
        """Assemble an OpenAI-compatible request payload (chat or responses)."""
        if not self.validate_model_name(model_name):
            raise ValueError(
                f"Model SQ{model_name}SQ not in allowed models list. Allowed models: {self.allowed_models}".replace(
                    "SQ", chr(39)
                )
            )

        capabilities: ModelCapabilities | None
        try:
            capabilities = self.get_capabilities(model_name)
        except Exception as exc:
            logging.debug(f"Falling back to generic capabilities for {model_name}: {exc}")
            capabilities = None

        if capabilities:
            effective_temperature = capabilities.get_effective_temperature(temperature)
            if effective_temperature is not None and effective_temperature != temperature:
                logging.debug(
                    f"Adjusting temperature from {temperature} to {effective_temperature} for model {model_name}"
                )
        else:
            effective_temperature = temperature

        if effective_temperature is not None:
            self.validate_parameters(model_name, effective_temperature)

        resolved_model = self._resolve_model_name(model_name)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, Any]] = []
        user_content.append({"type": "text", "text": prompt})

        if images and capabilities and capabilities.supports_images:
            for image_path in images:
                try:
                    image_content = self._process_image(image_path)
                    if image_content:
                        user_content.append(image_content)
                except Exception as e:
                    logging.warning(f"Failed to process image {image_path}: {e}")
                    continue
        elif images and (not capabilities or not capabilities.supports_images):
            logging.warning(f"Model {resolved_model} does not support images, ignoring {len(images)} image(s)")

        if len(user_content) == 1:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_content})

        use_responses_api = False
        if capabilities is not None:
            use_responses_api = getattr(capabilities, "use_openai_response_api", False)
        else:
            static_capabilities = self.get_all_model_capabilities().get(resolved_model)
            if static_capabilities is not None:
                use_responses_api = getattr(static_capabilities, "use_openai_response_api", False)

        if use_responses_api:
            input_messages: list[dict[str, Any]] = []
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                if role == "system":
                    input_messages.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
                elif role == "user":
                    if isinstance(content, str):
                        input_messages.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
                    else:
                        input_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    input_messages.append({"role": "assistant", "content": [{"type": "output_text", "text": content}]})

            effort = "medium"
            if capabilities and capabilities.default_reasoning_effort:
                effort = capabilities.default_reasoning_effort
            completion_params: dict[str, Any] = {
                "model": resolved_model,
                "input": input_messages,
                "reasoning": {"effort": effort},
            }
            if self.get_provider_type() != ProviderType.OPENROUTER:
                completion_params["store"] = True
            if max_output_tokens:
                completion_params["max_completion_tokens"] = max_output_tokens
            return {
                "endpoint": "responses",
                "model": resolved_model,
                "params": completion_params,
                "capabilities": capabilities,
            }

        completion_params = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
        }
        supports_sampling = effective_temperature is not None
        if supports_sampling:
            completion_params["temperature"] = effective_temperature
        if max_output_tokens and supports_sampling:
            completion_params["max_tokens"] = max_output_tokens
        for key, value in kwargs.items():
            if key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream"]:
                if not supports_sampling and key in ["top_p", "frequency_penalty", "presence_penalty", "stream"]:
                    continue
                completion_params[key] = value
        return {
            "endpoint": "chat",
            "model": resolved_model,
            "params": completion_params,
            "capabilities": capabilities,
        }

    def _call_api(self, request: dict[str, Any]) -> Any:
        """Invoke the OpenAI SDK using either chat or responses endpoint."""
        params = request["params"]
        if request.get("endpoint") == "responses":
            import json

            sanitized = self._sanitize_for_logging(params)
            logging.info(f"o3-pro API request (sanitized): {json.dumps(sanitized, indent=2, ensure_ascii=False)}")
            return self.client.responses.create(**params)
        return self.client.chat.completions.create(**params)

    def _parse_response(self, raw: Any, *, model_name: str, request: dict[str, Any]) -> ModelResponse:
        """Convert an OpenAI SDK response into a ModelResponse."""
        resolved_model = request["model"]
        if request.get("endpoint") == "responses":
            content = self._safe_extract_output_text(raw)
            usage: dict[str, int] = {}
            if hasattr(raw, "usage"):
                usage = self._extract_usage(raw)
            elif hasattr(raw, "input_tokens") and hasattr(raw, "output_tokens"):
                input_tokens = getattr(raw, "input_tokens", 0) or 0
                output_tokens = getattr(raw, "output_tokens", 0) or 0
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            return ModelResponse(
                content=content,
                usage=usage,
                model_name=resolved_model,
                friendly_name=self.FRIENDLY_NAME,
                provider=self.get_provider_type(),
                metadata={
                    "model": getattr(raw, "model", resolved_model),
                    "id": getattr(raw, "id", ""),
                    "created": getattr(raw, "created_at", 0),
                    "endpoint": "responses",
                },
            )
        content = raw.choices[0].message.content
        usage = self._extract_usage(raw)
        return ModelResponse(
            content=content,
            usage=usage,
            model_name=resolved_model,
            friendly_name=self.FRIENDLY_NAME,
            provider=self.get_provider_type(),
            metadata={
                "finish_reason": raw.choices[0].finish_reason,
                "model": raw.model,
                "id": raw.id,
                "created": raw.created,
            },
        )

    def _retry_log_prefix(self, request: dict[str, Any], model_name: str) -> str:
        resolved = request.get("model") or model_name
        if request.get("endpoint") == "responses":
            return "responses endpoint"
        return f"{self.FRIENDLY_NAME} API ({resolved})"

    def validate_parameters(self, model_name: str, temperature: float, **kwargs) -> None:
        """Validate model parameters."""
        try:
            capabilities = self.get_capabilities(model_name)
            if hasattr(capabilities, "_is_generic"):
                logging.debug(
                    f"Using generic parameter validation for {model_name}. Actual model constraints may differ."
                )
            super().validate_parameters(model_name, temperature, **kwargs)
        except Exception as e:
            logging.warning(f"Parameter validation limited for {model_name}: {e}")

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from OpenAI response."""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage["input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
            usage["output_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
            usage["total_tokens"] = getattr(response.usage, "total_tokens", 0) or 0
        return usage

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens using OpenAI-compatible tokenizer tables when available."""
        resolved_model = self._resolve_model_name(model_name)
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(resolved_model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except (ImportError, Exception) as exc:
            logging.debug("tiktoken unavailable for %s: %s", resolved_model, exc)
        return super().count_tokens(text, model_name)

    def _is_error_retryable(self, error: Exception) -> bool:
        """Determine if an error should be retried based on structured error codes."""
        error_str = str(error).lower()
        if "429" in error_str:
            error_type = None
            error_code = None
            try:
                import ast
                import json
                import re

                json_match = re.search(r"\{.*\}", str(error))
                if json_match:
                    json_like_str = json_match.group(0)
                    try:
                        error_data = ast.literal_eval(json_like_str)
                    except (ValueError, SyntaxError):
                        json_str = json_like_str.replace(chr(39), chr(34))
                        error_data = json.loads(json_str)
                    if "error" in error_data:
                        error_info = error_data["error"]
                        error_type = error_info.get("type")
                        error_code = error_info.get("code")
            except (json.JSONDecodeError, ValueError, SyntaxError, AttributeError):
                response_obj = getattr(error, "response", None)
                json_method = getattr(response_obj, "json", None)
                if callable(json_method):
                    try:
                        response_data = json_method()
                        if "error" in response_data:
                            error_info = response_data["error"]
                            error_type = error_info.get("type")
                            error_code = error_info.get("code")
                    except Exception:
                        pass
            if error_type == "tokens":
                logging.debug(f"Non-retryable 429: token-related error (type={error_type}, code={error_code})")
                return False
            elif error_code in ["invalid_request_error", "context_length_exceeded"]:
                logging.debug(f"Non-retryable 429: permanent failure (type={error_type}, code={error_code})")
                return False
            else:
                logging.debug(f"Retryable 429: rate limiting (type={error_type}, code={error_code})")
                return True
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "unavailable",
            "retry",
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
        """Process an image for OpenAI-compatible API."""
        try:
            if image_path.startswith("data:"):
                validate_image(image_path)
                return {"type": "image_url", "image_url": {"url": image_path}}
            else:
                image_bytes, mime_type = validate_image(image_path)
                import base64

                image_data = base64.b64encode(image_bytes).decode()
                logging.debug(f"Processing image SQ{image_path}SQ as MIME type SQ{mime_type}SQ".replace("SQ", chr(39)))
                data_url = f"data:{mime_type};base64,{image_data}"
                return {"type": "image_url", "image_url": {"url": data_url}}
        except ValueError as e:
            logging.warning(str(e))
            return None
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None
