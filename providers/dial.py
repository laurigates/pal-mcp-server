"""DIAL (Data & AI Layer) model provider implementation."""

import asyncio
import logging
import threading
from typing import Any, ClassVar

from utils.env import get_env

from .openai_compatible import OpenAICompatibleProvider
from .registries.dial import DialModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class DIALModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Client for the DIAL aggregation service."""

    FRIENDLY_NAME = "DIAL"

    REGISTRY_CLASS = DialModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    MAX_RETRIES = 4
    RETRY_DELAYS = [1.0, 3.0, 5.0, 8.0]

    def __init__(self, api_key: str, **kwargs):
        """Initialize DIAL provider with API key and host."""
        self._ensure_registry()
        dial_host = kwargs.get("base_url") or get_env("DIAL_API_HOST") or "https://core.dialx.ai"
        if not dial_host.endswith("/openai"):
            dial_host = f"{dial_host.rstrip(chr(47))}/openai"
        kwargs["base_url"] = dial_host
        self.api_version = get_env("DIAL_API_VERSION", "2024-12-01-preview") or "2024-12-01-preview"
        self.DEFAULT_HEADERS = {"Api-Key": api_key}
        self._dial_api_key = api_key
        super().__init__("placeholder-not-used", **kwargs)
        self._deployment_clients = {}
        self._client_lock = threading.Lock()
        import httpx

        def remove_auth_header(request):
            """Remove Authorization header that OpenAI client adds."""
            headers_to_remove = []
            for header_name in request.headers:
                if header_name.lower() == "authorization":
                    headers_to_remove.append(header_name)
            for header_name in headers_to_remove:
                del request.headers[header_name]

        self._http_client = httpx.AsyncClient(
            timeout=self.timeout_config,
            verify=True,
            follow_redirects=True,
            headers=self.DEFAULT_HEADERS.copy(),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            event_hooks={"request": [remove_auth_header]},
        )
        logger.info(f"Initialized DIAL provider with host: {dial_host} and api-version: {self.api_version}")

    @classmethod
    def from_env(cls) -> "DIALModelProvider | None":
        """Construct a provider from environment variables.

        Reads ``DIAL_API_KEY`` and rejects the documented placeholder
        value. Host (``DIAL_API_HOST``) and version (``DIAL_API_VERSION``)
        are honoured by ``__init__`` directly.

        Returns:
            A configured provider instance, or ``None`` when the API key is
            missing or set to the placeholder string.
        """
        api_key = get_env("DIAL_API_KEY")
        if not api_key or api_key == "your_dial_api_key_here":
            return None
        return cls(api_key=api_key)

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.DIAL

    def _get_deployment_client(self, deployment: str):
        """Get or create a cached client for a specific deployment."""
        if deployment in self._deployment_clients:
            return self._deployment_clients[deployment]
        with self._client_lock:
            if deployment not in self._deployment_clients:
                from openai import AsyncOpenAI

                base_url = str(self.client.base_url)
                if base_url.endswith("/"):
                    base_url = base_url[:-1]
                if base_url.endswith("/openai"):
                    base_url = base_url[:-7]
                deployment_url = f"{base_url}/openai/deployments/{deployment}"
                self._deployment_clients[deployment] = AsyncOpenAI(
                    api_key="placeholder-not-used",
                    base_url=deployment_url,
                    http_client=self._http_client,
                    default_query={"api-version": self.api_version},
                )
        return self._deployment_clients[deployment]

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
        """Assemble a DIAL chat-completions request."""
        if not self.validate_model_name(model_name):
            raise ValueError(
                f"Model SQ{model_name}SQ not in allowed models list. Allowed models: {self.allowed_models}".replace(
                    "SQ", chr(39)
                )
            )
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_message_content: list[dict[str, Any]] = []
        if prompt:
            user_message_content.append({"type": "text", "text": prompt})
        if images and capabilities.supports_images:
            for img_path in images:
                processed_image = self._process_image(img_path)
                if processed_image:
                    user_message_content.append(processed_image)
        elif images:
            logger.warning(f"Model {model_name} does not support images, ignoring {len(images)} image(s)")
        if len(user_message_content) == 1 and user_message_content[0]["type"] == "text":
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": user_message_content})

        resolved_model = self._resolve_model_name(model_name)
        completion_params: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
        }
        supports_temperature = capabilities.supports_temperature
        if supports_temperature:
            completion_params["temperature"] = temperature
        if max_output_tokens and supports_temperature:
            completion_params["max_tokens"] = max_output_tokens
        for key, value in kwargs.items():
            if key in ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop", "stream"]:
                if not supports_temperature and key in ["top_p", "frequency_penalty", "presence_penalty", "stream"]:
                    continue
                completion_params[key] = value
        return {
            "model": resolved_model,
            "requested_model_name": model_name,
            "params": completion_params,
        }

    async def _call_api(self, request: dict[str, Any]) -> Any:
        """Invoke DIAL deployment-specific chat.completions endpoint."""
        deployment_client = self._get_deployment_client(request["model"])
        return await deployment_client.chat.completions.create(**request["params"])

    def _parse_response(self, raw: Any, *, model_name: str, request: dict[str, Any]) -> ModelResponse:
        """Convert DIAL response into ModelResponse."""
        content = raw.choices[0].message.content
        usage = self._extract_usage(raw)
        return ModelResponse(
            content=content,
            usage=usage,
            model_name=request.get("requested_model_name", model_name),
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
        return f"DIAL API ({resolved})"

    def _wrap_api_failure(
        self,
        exc: Exception,
        *,
        request: dict[str, Any],
        model_name: str,
        attempts: int,
    ) -> Exception:
        """DIAL historically raised ValueError after retry exhaustion."""
        resolved = request.get("model") or model_name
        if attempts == 1:
            return ValueError(f"DIAL API error for model {resolved}: {exc}")
        return ValueError(f"DIAL API error for model {resolved} after {attempts} attempts: {exc}")

    def close(self) -> None:
        """Clean up HTTP clients when provider is closed."""
        logger.info("Closing DIAL provider HTTP clients...")
        self._deployment_clients.clear()

        async def _aclose_all() -> None:
            if hasattr(self, "_http_client"):
                try:
                    await self._http_client.aclose()
                    logger.debug("Closed shared HTTP client")
                except Exception as e:
                    logger.warning(f"Error closing shared HTTP client: {e}")
            if hasattr(self, "_client") and self._client and hasattr(self._client, "aclose"):
                try:
                    await self._client.aclose()
                    logger.debug("Closed superclass AsyncOpenAI client")
                except Exception as e:
                    logger.warning(f"Error closing superclass AsyncOpenAI client: {e}")

        try:
            asyncio.run(_aclose_all())
        except RuntimeError:
            # Event loop is already running (e.g. called from within async context);
            # schedule the cleanup as a best-effort fire-and-forget.
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(_aclose_all())
            except Exception as e:
                logger.warning(f"Could not schedule async cleanup: {e}")
