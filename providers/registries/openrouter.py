"""OpenRouter model registry for managing model configurations and aliases."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from typing import Any

from utils.env import get_env

from ..shared import ModelCapabilities, ProviderType
from . import catalog
from .base import CAPABILITY_FIELD_NAMES, CapabilityModelRegistry

logger = logging.getLogger(__name__)

_ENV_VAR_NAME = "OPENROUTER_MODELS_CONFIG_PATH"
_DEFAULT_FILENAME = "openrouter_models.json"

#: The newest catalog this process holds, as ``(payload, fetched_at, origin)``
#: with origin "live" or "cache". Read once from the cache file on first use,
#: then replaced by a successful refresh, so constructing a registry (several
#: call sites build a fresh one per lookup) never re-reads the 1 MB file.
_catalog_state: tuple[Any, float, str] | None = None
_catalog_state_loaded = False
_state_lock = threading.Lock()

#: Registries that take catalog facts; a refresh reloads each one in place.
_dynamic_registries: weakref.WeakSet[OpenRouterModelRegistry] = weakref.WeakSet()


def _current_catalog() -> tuple[Any, float, str] | None:
    global _catalog_state, _catalog_state_loaded
    with _state_lock:
        if not _catalog_state_loaded:
            _catalog_state_loaded = True
            cached = catalog.read_cache(_DEFAULT_FILENAME)
            if cached is not None:
                _catalog_state = (cached[0], cached[1], "cache")
        return _catalog_state


def _reset_catalog_state() -> None:
    """Forget the in-process catalog so the next registry re-reads the cache (tests)."""
    global _catalog_state, _catalog_state_loaded
    with _state_lock:
        _catalog_state = None
        _catalog_state_loaded = False


class OpenRouterModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/openrouter_models.json``.

    Unless ``PAL_DYNAMIC_REGISTRY=0`` is set or the file is pinned (an explicit
    ``config_path`` or ``OPENROUTER_MODELS_CONFIG_PATH``), the bundled entries
    are laid over the OpenRouter catalog: the catalog supplies limits and
    capability flags, the bundled file everything curated (ADR-0001).
    """

    def __init__(self, config_path: str | None = None) -> None:
        pinned = bool(config_path or get_env(_ENV_VAR_NAME))
        self._dynamic = catalog.dynamic_registry_enabled() and not pinned
        #: ``(origin, fetched_at)`` of the catalog behind the current maps:
        #: origin is "live", "cache", "bundled" or "pinned".
        self.registry_source: tuple[str, float | None] = ("pinned", None) if pinned else ("bundled", None)
        #: Lowercased names of curated entries the catalog no longer lists.
        #: ADR-0001 decision 7: a call by such a name fails as unavailable
        #: instead of reaching OpenRouter and surfacing its 404.
        self.withdrawn: frozenset[str] = frozenset()
        super().__init__(
            env_var_name=_ENV_VAR_NAME,
            default_filename=_DEFAULT_FILENAME,
            provider=ProviderType.OPENROUTER,
            friendly_prefix="OpenRouter ({model})",
            config_path=config_path,
        )
        if self._dynamic:
            _dynamic_registries.add(self)

    def _load_config_data(self) -> dict:
        data = super()._load_config_data()
        self.withdrawn = frozenset()
        if not self._dynamic:
            return data
        state = _current_catalog()
        if state is None:
            self.registry_source = ("bundled", None)
            return data
        payload, fetched_at, origin = state
        try:
            records = catalog.validate_openrouter_catalog(payload)
            models, summary = catalog.merge_openrouter_catalog(list(data.get("models", [])), records)
        except ValueError as exc:
            logger.warning("Ignoring OpenRouter catalog (%s); using bundled registry: %s", origin, exc)
            self.registry_source = ("bundled", None)
            return data
        if summary["withdrawn"]:
            logger.info(
                "OpenRouter catalog no longer lists %d curated model(s); disabled: %s",
                len(summary["withdrawn"]),
                ", ".join(summary["withdrawn"]),
            )
        self.registry_source = (origin, fetched_at)
        self.withdrawn = frozenset(name.lower() for name in summary["absent"])
        return {**data, "models": models}

    def is_withdrawn(self, model_name: str) -> bool:
        """True when ``model_name`` is a curated entry the catalog no longer lists."""
        return model_name.lower() in self.withdrawn

    def describe_source(self) -> str:
        """One line for ``listmodels``: where the current entries came from."""
        origin, fetched_at = self.registry_source
        if origin == "pinned":
            return "pinned file (catalog refresh off)"
        if origin == "bundled":
            if not self._dynamic:
                return "bundled file (PAL_DYNAMIC_REGISTRY=0)"
            return "bundled file (no catalog fetched yet)"
        assert fetched_at is not None
        stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(fetched_at))
        return f"{origin} OpenRouter catalog, fetched {stamp} ({catalog.describe_age(fetched_at)} ago)"

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        provider_override = entry.get("provider")
        if isinstance(provider_override, str):
            entry_provider = ProviderType(provider_override.lower())
        elif isinstance(provider_override, ProviderType):
            entry_provider = provider_override
        else:
            entry_provider = ProviderType.OPENROUTER

        if entry_provider == ProviderType.CUSTOM:
            entry.setdefault("friendly_name", f"Custom ({entry['model_name']})")
        else:
            entry.setdefault("friendly_name", f"OpenRouter ({entry['model_name']})")

        filtered = {k: v for k, v in entry.items() if k in CAPABILITY_FIELD_NAMES}
        filtered.setdefault("provider", entry_provider)
        capability = ModelCapabilities(**filtered)
        return capability, {}


def openrouter_catalog_refresh_enabled() -> bool:
    return catalog.dynamic_registry_enabled() and not get_env(_ENV_VAR_NAME)


def refresh_openrouter_catalog(url: str = catalog.OPENROUTER_CATALOG_URL) -> bool:
    """Fetch the catalog, cache it and reload every dynamic registry.

    Returns ``True`` when fresh data was applied. A failed fetch or an invalid
    payload keeps the previous data and logs a warning; an unwritable cache
    still applies the fetched data to this process.
    """
    global _catalog_state, _catalog_state_loaded
    try:
        payload = catalog.fetch_json(url)
        catalog.validate_openrouter_catalog(payload)
    except Exception as exc:
        logger.warning("OpenRouter catalog refresh failed; keeping the previous registry: %s", exc)
        return False

    fetched_at = time.time()
    try:
        path = catalog.write_cache(_DEFAULT_FILENAME, payload, fetched_at)
        logger.debug("Cached OpenRouter catalog at %s", path)
    except OSError as exc:
        logger.warning("Could not write OpenRouter catalog cache: %s", exc)

    with _state_lock:
        _catalog_state = (payload, fetched_at, "live")
        _catalog_state_loaded = True

    for registry in list(_dynamic_registries):
        try:
            registry.reload()
        except Exception as exc:  # a broken bundled file is reported at startup, not here
            logger.warning("Reloading an OpenRouter registry after refresh failed: %s", exc)
    logger.info("OpenRouter catalog refreshed: %d models", len(payload["data"]))
    return True


def start_openrouter_catalog_refresh() -> threading.Thread | None:
    """Run ``refresh_openrouter_catalog`` on a daemon thread, or return ``None`` when off."""
    if not openrouter_catalog_refresh_enabled():
        return None
    thread = threading.Thread(target=refresh_openrouter_catalog, name="openrouter-catalog-refresh", daemon=True)
    thread.start()
    return thread
