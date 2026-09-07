"""Drift guards for the declarative provider metadata (issue #60).

These tests make "add a provider class but forget to declare its metadata"
fail loudly instead of silently producing a provider that half the codebase
cannot see.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

from providers.base import ModelProvider
from providers.registry import PROVIDER_CLASS_BY_TYPE, REGISTERED_PROVIDER_CLASSES, ModelProviderRegistry
from providers.shared import ProviderType

pytestmark = pytest.mark.no_mock_provider

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROVIDERS_DIR = _REPO_ROOT / "providers"

#: Modules in providers/ that are infrastructure, not concrete providers.
_INFRASTRUCTURE_MODULES = {
    "providers.base",
    "providers.openai_compatible",
    "providers.registry",
    "providers.registry_provider_mixin",
}

#: Concrete providers that deliberately declare no ``API_KEY_PLACEHOLDER``,
#: mapped to the reason. An entry is a reviewed waiver, not a default: the
#: value of the map is that adding one shows up in a diff and has to be
#: justified. ``CustomProvider`` sat outside the guard by omission rather than
#: by decision for as long as it existed, which is issue #115.
_PLACEHOLDER_WAIVERS: dict[str, str] = {}


def _concrete_provider_classes() -> list[type[ModelProvider]]:
    """Every ``ModelProvider`` subclass defined in a ``providers/*.py`` module.

    Walked from the module tree rather than read off a hand-kept list, so a
    provider added tomorrow is covered without anyone remembering to edit this
    file. Classes are attributed by ``__module__`` so an imported base class
    is not counted against the module that imports it.
    """
    discovered: list[type[ModelProvider]] = []
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        module_name = f"providers.{path.stem}"
        if module_name in _INFRASTRUCTURE_MODULES or path.stem.startswith("_"):
            continue
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ModelProvider) and obj.__module__ == module_name:
                discovered.append(obj)
    return discovered


def test_every_provider_type_has_exactly_one_registered_class():
    """PROVIDER_CLASS_BY_TYPE must be a total function over ProviderType."""
    assert sorted(pt.value for pt in PROVIDER_CLASS_BY_TYPE) == sorted(pt.value for pt in ProviderType)
    assert len(PROVIDER_CLASS_BY_TYPE) == len(REGISTERED_PROVIDER_CLASSES)


def test_priority_order_is_derived_from_the_class_list():
    """PROVIDER_PRIORITY_ORDER is list position, not a second hand-written table."""
    assert ModelProviderRegistry.PROVIDER_PRIORITY_ORDER == [
        provider_cls.provider_type() for provider_cls in REGISTERED_PROVIDER_CLASSES
    ]


def test_openrouter_sorts_last():
    """OpenRouter fabricates capabilities for any name containing '/', so it
    must stay last or it hijacks routing for every other provider."""
    assert REGISTERED_PROVIDER_CLASSES[-1].provider_type() is ProviderType.OPENROUTER


def test_every_registered_class_declares_its_identity():
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        assert "PROVIDER_TYPE" in vars(provider_cls), f"{provider_cls.__name__} must declare PROVIDER_TYPE"
        assert "FRIENDLY_NAME" in vars(provider_cls), (
            f"{provider_cls.__name__} inherits FRIENDLY_NAME instead of declaring it"
        )
        assert vars(provider_cls)["FRIENDLY_NAME"] != "OpenAI Compatible", provider_cls.__name__
        assert provider_cls.gating_env_vars(), f"{provider_cls.__name__} declares no activation gate"


def test_every_provider_module_with_from_env_has_a_registered_class():
    """A new providers/<name>.py must be added to REGISTERED_PROVIDER_CLASSES."""
    registered_modules = {provider_cls.__module__ for provider_cls in REGISTERED_PROVIDER_CLASSES}

    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        module = f"providers.{path.stem}"
        if module in _INFRASTRUCTURE_MODULES or path.stem.startswith("_"):
            continue
        if "def from_env(" not in path.read_text(encoding="utf-8"):
            continue
        assert module in registered_modules, (
            f"{module} defines from_env() but is not in providers.registry._build_registered_provider_classes()"
        )


def test_no_undeclared_placeholder_literal_survives_in_providers():
    """Every ``your_*_here`` literal under providers/ must be a declared
    API_KEY_PLACEHOLDER - no provider may keep a private copy."""
    declared = {
        provider_cls.API_KEY_PLACEHOLDER
        for provider_cls in REGISTERED_PROVIDER_CLASSES
        if provider_cls.API_KEY_PLACEHOLDER
    }
    for path in sorted(_PROVIDERS_DIR.rglob("*.py")):
        for literal in re.findall(r'"(your_[a-z0-9_]+_here)"', path.read_text(encoding="utf-8")):
            assert literal in declared, f"{path}: undeclared placeholder {literal!r}"


def test_the_module_walk_finds_every_registered_provider():
    """Control for the two tests below.

    They iterate ``_concrete_provider_classes()``; an empty or partial walk
    would let them pass while asserting nothing. Anchoring it to the
    registered roster means a broken walk fails here instead of going quiet.

    Compared by qualified name, not identity: a test that ``importlib.reload``s
    a provider module leaves ``REGISTERED_PROVIDER_CLASSES`` holding the
    pre-reload class object, so an identity comparison fails on run order
    rather than on coverage (observed here for OpenRouterProvider).
    """
    discovered = {f"{cls.__module__}.{cls.__qualname__}" for cls in _concrete_provider_classes()}
    assert discovered, "the providers/ module walk found no provider classes"
    registered = {f"{cls.__module__}.{cls.__qualname__}" for cls in REGISTERED_PROVIDER_CLASSES}
    assert registered <= discovered, sorted(registered - discovered)


def test_every_api_key_provider_declares_a_placeholder_or_is_waived():
    """Issue #115: ``CustomProvider`` was the one provider outside the guard.

    ``ModelProvider.api_key_from_env`` rejects a key equal to the class's
    declared ``API_KEY_PLACEHOLDER``, so a provider that declares none accepts
    a scaffolded ``.env`` value as a real credential and then fails on first
    call instead of reading as unconfigured.
    """
    for provider_cls in _concrete_provider_classes():
        name = provider_cls.__name__
        if not provider_cls.API_KEY_ENV:
            assert name not in _PLACEHOLDER_WAIVERS, f"{name} reads no API key, so its waiver is dead"
            continue
        if name in _PLACEHOLDER_WAIVERS:
            assert _PLACEHOLDER_WAIVERS[name].strip(), f"{name}'s placeholder waiver must state a reason"
            continue
        assert vars(provider_cls).get("API_KEY_PLACEHOLDER"), (
            f"{name} reads {provider_cls.API_KEY_ENV} but declares no API_KEY_PLACEHOLDER. "
            "Declare the value .env.example ships, or add a reasoned entry to _PLACEHOLDER_WAIVERS."
        )


def test_placeholder_waivers_name_a_live_provider():
    """A waiver for a deleted or renamed provider must not linger unnoticed."""
    known = {provider_cls.__name__ for provider_cls in _concrete_provider_classes()}
    assert set(_PLACEHOLDER_WAIVERS) <= known, sorted(set(_PLACEHOLDER_WAIVERS) - known)


def test_every_declared_placeholder_is_documented_in_env_example():
    """The guard only fires on the exact string the template ships.

    A placeholder declared in code but absent from ``.env.example`` (or drifted
    from it) is a guard that matches nothing anybody will ever have set.
    """
    template = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        placeholder = provider_cls.API_KEY_PLACEHOLDER
        if not placeholder:
            continue
        assignment = f"{provider_cls.API_KEY_ENV}={placeholder}"
        assert assignment in template, f"{provider_cls.__name__}: .env.example does not ship {assignment!r}"


def test_no_provider_module_imports_the_registry_at_module_scope():
    """The registry imports every provider module; a reverse module-scope
    import would reintroduce the import cycle the deferral used to hide."""
    for path in sorted(_PROVIDERS_DIR.glob("*.py")):
        if path.stem in {"registry", "__init__"}:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(
                ("from .registry import", "from providers.registry import", "import providers.registry")
            ):
                raise AssertionError(f"{path.name} imports providers.registry at module scope: {line!r}")


def test_every_provider_type_has_a_resolvable_allowlist_env_var():
    """No provider may be un-restrictable.

    OPENCODE_GO, AZURE and CUSTOM were silently unrestrictable because
    ModelRestrictionService.ENV_VARS covered 5 of 8 provider types - the
    root cause of issue #66's listing-time leak.
    """
    from utils.model_restrictions import ModelRestrictionService

    for provider_type in ProviderType:
        env_vars = ModelRestrictionService.env_vars_for(provider_type)
        assert env_vars, provider_type
        assert all(var.endswith("_ALLOWED_MODELS") for var in env_vars), (provider_type, env_vars)
