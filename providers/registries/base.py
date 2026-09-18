"""Shared infrastructure for JSON-backed model registries."""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from utils.env import get_env

from ..shared import ModelCapabilities, ProviderType, TemperatureConstraint

logger = logging.getLogger(__name__)


CAPABILITY_FIELD_NAMES = {field.name for field in fields(ModelCapabilities)}

#: Bound for load_or_raise() below so it returns the concrete subclass
#: (e.g. CustomEndpointModelRegistry) rather than the base class -- Self
#: isn't used since it requires Python 3.11+ and this project supports 3.10.
_RegistryT = TypeVar("_RegistryT", bound="CustomModelRegistryBase")


class CustomModelRegistryBase:
    """Load and expose capability metadata from a JSON manifest."""

    def __init__(
        self,
        *,
        env_var_name: str,
        default_filename: str,
        config_path: str | None = None,
    ) -> None:
        self._env_var_name = env_var_name
        self._default_filename = default_filename
        self._use_resources = False
        self._resource_package = "conf"
        self._default_path = Path(__file__).resolve().parents[3] / "conf" / default_filename

        if config_path:
            self.config_path = Path(config_path)
        else:
            env_path = get_env(env_var_name)
            if env_path:
                self.config_path = Path(env_path)
            else:
                try:
                    resource = importlib.resources.files(self._resource_package).joinpath(default_filename)
                    if hasattr(resource, "read_text"):
                        self._use_resources = True
                        self.config_path = None
                    else:
                        raise AttributeError("resource accessor not available")
                except Exception:
                    self.config_path = Path(__file__).resolve().parents[3] / "conf" / default_filename

        self.alias_map: dict[str, str] = {}
        self.model_map: dict[str, ModelCapabilities] = {}
        self._extras: dict[str, dict] = {}
        #: Set when the backing file exists but fails to parse or validate --
        #: as opposed to simply being absent, which stays ``None`` (see
        #: ``_load_config_data`` and ``reload``). A ``(config_path, reason)``
        #: pair rather than an exception instance so recording it never
        #: requires importing ``providers.registry`` (circular, see
        #: ``raise_if_config_error``). Populated but not acted on here: some
        #: registries (e.g. ``OpenRouterModelRegistry``) intentionally keep
        #: today's lenient degrade-to-empty behaviour, so this is opt-in --
        #: callers that want issue #130's "configured but broken" treatment
        #: use ``load_or_raise``/``raise_if_config_error`` instead of the bare
        #: constructor.
        self.config_error: tuple[str, str] | None = None

    def reload(self) -> None:
        data = self._load_config_data()
        try:
            configs = [config for config in self._parse_models(data) if config is not None]
            self._build_maps(configs)
        except ValueError as exc:
            # Schema violation (_convert_entry) or duplicate alias
            # (_build_maps) -- record it like the JSON-syntax case below, but
            # re-raise the *same* ValueError unchanged so every existing
            # caller (including registries that don't opt into
            # ModelRegistryConfigError) keeps seeing exactly what it sees
            # today. Tag the exception itself so a caller that only has the
            # exception in hand (construction raised before it got a
            # registry instance back) can still recover the source path.
            self.config_error = (self._config_source(), str(exc))
            # setattr rather than exc.registry_config_error = ... so a type
            # checker doesn't need ValueError to declare this attribute.
            setattr(exc, "registry_config_error", self.config_error)  # noqa: B010
            raise

    def _config_source(self) -> str:
        """Best-effort path to this registry's backing file, for error messages."""
        if self.config_path is not None:
            return str(self.config_path)
        try:
            resource = importlib.resources.files(self._resource_package).joinpath(self._default_filename)
            return str(resource)
        except Exception:
            return self._default_filename

    def raise_if_config_error(self) -> None:
        """Escalate a recorded-but-swallowed parse failure to ``ModelRegistryConfigError``.

        A no-op when nothing was recorded. Imported lazily: ``providers.registry``
        finishes building ``REGISTERED_PROVIDER_CLASSES`` -- which imports every
        provider module, which imports every ``providers.registries`` module --
        before ``providers.registries.base`` would otherwise finish importing,
        so a module-level import here is circular.
        """
        if self.config_error is None:
            return
        from providers.registry import ModelRegistryConfigError

        config_path, reason = self.config_error
        raise ModelRegistryConfigError(config_path, reason)

    @classmethod
    def load_or_raise(cls: type[_RegistryT], *args, **kwargs) -> _RegistryT:
        """Construct this registry, raising ``ModelRegistryConfigError`` for a broken file.

        Equivalent to calling the constructor directly except that a
        malformed or schema-invalid backing file raises the dedicated,
        file-naming exception (issue #130) instead of degrading to an empty
        registry or a generic ``ValueError``. Any other failure (e.g. a
        subclass constructor rejecting bad arguments) propagates unchanged.
        """
        try:
            instance = cls(*args, **kwargs)
        except ValueError as exc:
            registry_error = getattr(exc, "registry_config_error", None)
            if registry_error is None:
                raise
            from providers.registry import ModelRegistryConfigError

            config_path, reason = registry_error
            raise ModelRegistryConfigError(config_path, reason) from exc
        instance.raise_if_config_error()
        return instance

    def list_models(self) -> list[str]:
        return list(self.model_map.keys())

    def list_aliases(self) -> list[str]:
        return list(self.alias_map.keys())

    def resolve(self, name_or_alias: str) -> ModelCapabilities | None:
        key = name_or_alias.lower()
        canonical = self.alias_map.get(key)
        if canonical:
            return self.model_map.get(canonical)

        for model_name in self.model_map:
            if model_name.lower() == key:
                return self.model_map[model_name]
        return None

    def get_capabilities(self, name_or_alias: str) -> ModelCapabilities | None:
        return self.resolve(name_or_alias)

    def get_entry(self, model_name: str) -> dict | None:
        return self._extras.get(model_name)

    def get_model_config(self, model_name: str) -> ModelCapabilities | None:
        """Backwards-compatible accessor for registries expecting this helper."""

        return self.model_map.get(model_name) or self.resolve(model_name)

    def iter_entries(self) -> Iterable[tuple[str, ModelCapabilities, dict]]:
        for model_name, capability in self.model_map.items():
            yield model_name, capability, self._extras.get(model_name, {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_config_data(self) -> dict:
        if self._use_resources:
            resource = importlib.resources.files(self._resource_package).joinpath(self._default_filename)
            try:
                if hasattr(resource, "read_text"):
                    config_text = resource.read_text(encoding="utf-8")
                else:  # pragma: no cover - legacy Python fallback
                    with resource.open("r", encoding="utf-8") as handle:
                        config_text = handle.read()
            except FileNotFoundError:
                logger.debug("Packaged %s not found", self._default_filename)
                return {"models": []}
            except Exception as exc:
                logger.warning("Failed to read packaged %s: %s", self._default_filename, exc)
                return {"models": []}

            try:
                data = json.loads(config_text)
            except json.JSONDecodeError as exc:
                # File is present but unparseable -- distinct from "absent"
                # above. Recorded, not raised: some registries (see
                # ``config_error``'s docstring) keep today's lenient degrade;
                # ``load_or_raise``/``raise_if_config_error`` escalate this
                # for the ones that opt in (issue #130).
                logger.warning("Invalid JSON in packaged %s: %s", self._default_filename, exc)
                self.config_error = (str(resource), f"Invalid JSON in model registry configuration ({exc})")
                return {"models": []}
            return data or {"models": []}

        if not self.config_path:
            raise FileNotFoundError("Registry configuration path is not set")

        if not self.config_path.exists():
            logger.debug("Model registry config not found at %s", self.config_path)
            if self.config_path == self._default_path:
                fallback = Path.cwd() / "conf" / self._default_filename
                if fallback != self.config_path and fallback.exists():
                    logger.debug("Falling back to %s", fallback)
                    self.config_path = fallback
                else:
                    return {"models": []}
            else:
                return {"models": []}

        # Parsed directly here (rather than via utils.file_utils.read_json_file,
        # which folds "malformed" and "absent" into the same None return) so
        # that distinction survives for the caller. This keeps
        # read_json_file's contract -- and its other caller,
        # clink/registry.py -- untouched.
        try:
            with open(self.config_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Invalid model registry configuration at %s: %s", self.config_path, exc)
            self.config_error = (str(self.config_path), f"Invalid model registry configuration ({exc})")
            return {"models": []}
        return data or {"models": []}

    @property
    def use_resources(self) -> bool:
        return self._use_resources

    def _parse_models(self, data: dict) -> Iterable[ModelCapabilities | None]:
        for raw in data.get("models", []):
            if not isinstance(raw, dict):
                continue
            yield self._convert_entry(raw)

    def _convert_entry(self, raw: dict) -> ModelCapabilities | None:
        entry = dict(raw)
        model_name = entry.get("model_name")
        if not model_name:
            return None

        aliases = entry.get("aliases")
        if isinstance(aliases, str):
            entry["aliases"] = [alias.strip() for alias in aliases.split(",") if alias.strip()]

        entry.setdefault("friendly_name", self._default_friendly_name(model_name))

        temperature_hint = entry.get("temperature_constraint")
        if isinstance(temperature_hint, str):
            entry["temperature_constraint"] = TemperatureConstraint.create(temperature_hint)
        elif temperature_hint is None:
            entry["temperature_constraint"] = TemperatureConstraint.create("range")

        if "max_tokens" in entry:
            raise ValueError(
                "`max_tokens` is no longer supported. Use `max_output_tokens` in your model configuration."
            )

        unknown_keys = set(entry.keys()) - CAPABILITY_FIELD_NAMES - self._extra_keys()
        if unknown_keys:
            raise ValueError("Unsupported fields in model configuration: " + ", ".join(sorted(unknown_keys)))

        capability, extras = self._finalise_entry(entry)
        capability.provider = self._provider_default()
        self._extras[capability.model_name] = extras or {}
        return capability

    def _default_friendly_name(self, model_name: str) -> str:
        return model_name

    def _extra_keys(self) -> set[str]:
        return set()

    def _provider_default(self) -> ProviderType:
        return ProviderType.OPENROUTER

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        return ModelCapabilities(**{k: v for k, v in entry.items() if k in CAPABILITY_FIELD_NAMES}), {}

    def _build_maps(self, configs: Iterable[ModelCapabilities]) -> None:
        alias_map: dict[str, str] = {}
        model_map: dict[str, ModelCapabilities] = {}

        for config in configs:
            if not config:
                continue
            model_map[config.model_name] = config

            model_name_lower = config.model_name.lower()
            if model_name_lower not in alias_map:
                alias_map[model_name_lower] = config.model_name

            for alias in config.aliases:
                alias_lower = alias.lower()
                if alias_lower in alias_map and alias_map[alias_lower] != config.model_name:
                    raise ValueError(
                        f"Duplicate alias '{alias}' found for models '{alias_map[alias_lower]}' and '{config.model_name}'"
                    )
                alias_map[alias_lower] = config.model_name

        self.alias_map = alias_map
        self.model_map = model_map


class CapabilityModelRegistry(CustomModelRegistryBase):
    """Registry that returns :class:`ModelCapabilities` objects with alias support."""

    def __init__(
        self,
        *,
        env_var_name: str,
        default_filename: str,
        provider: ProviderType,
        friendly_prefix: str,
        config_path: str | None = None,
    ) -> None:
        self._provider = provider
        self._friendly_prefix = friendly_prefix
        super().__init__(
            env_var_name=env_var_name,
            default_filename=default_filename,
            config_path=config_path,
        )
        self.reload()

    def _provider_default(self) -> ProviderType:
        return self._provider

    def _default_friendly_name(self, model_name: str) -> str:
        return self._friendly_prefix.format(model=model_name)

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        filtered = {k: v for k, v in entry.items() if k in CAPABILITY_FIELD_NAMES}
        filtered.setdefault("provider", self._provider_default())
        capability = ModelCapabilities(**filtered)
        return capability, {}
