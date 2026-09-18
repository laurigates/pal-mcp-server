"""Startup behaviour for a broken model registry (issue #130).

``configure_providers()`` used to wrap every provider's ``from_env()`` in a
blanket ``except Exception``, so a checked-in ``conf/*_models.json`` that
fails to parse or validate made the provider silently vanish -- reported
through the issue #116 "no providers configured" remedy exactly as if the
API key had never been set. That is the wrong fix for the operator: the key
*is* set, the config file is just broken.

The fix distinguishes the two conditions with a dedicated exception,
:class:`~providers.registry.ModelRegistryConfigError`: a provider's
``from_env()`` letting that propagate means "configured but broken" and must
fail startup naming the offending file, while every other exception (or a
plain ``None`` return) still means "not configured" and stays on the #116
path.

The classes below (``_FakeProviderBase`` and its two subclasses) pin the
contract at the boundary ``configure_providers()`` owns, with small stand-in
provider classes rather than a real provider -- a real provider's ``from_env``
chain (registry caching, optional-dependency imports, ...) is exactly the
kind of incidental machinery those tests should not depend on. Each stand-in
raises or returns exactly what a real provider's ``from_env()`` does now that
``providers/registries/base.py`` raises ``ModelRegistryConfigError`` for a
broken registry file.

The classes further down exercise that real wiring end to end: a real
registry class (``CustomEndpointModelRegistry``) reading a genuinely
malformed ``tmp_path`` file, and ``CustomProvider.from_env()`` /
``configure_providers()`` on top of it. See ``providers/registries/base.py``
for the mechanism -- ``config_error``/``raise_if_config_error()``/
``load_or_raise()`` -- and why it is opt-in: ``OpenRouterModelRegistry``
shares the same base class and its own tests
(``tests/test_openrouter_registry.py``) pin the pre-#130 lenient behaviour
(a malformed file degrades to an empty registry; a schema violation raises
plain ``ValueError``) for the bare constructor, so that behaviour could not
change unconditionally.
"""

import json

import pytest

import server
from providers.custom import CustomProvider
from providers.registries.custom import CustomEndpointModelRegistry
from providers.registry import ModelProviderRegistry, ModelRegistryConfigError
from providers.shared import ProviderType


class _FakeProviderBase:
    """Minimal stand-in satisfying what ``configure_providers()`` calls on a
    provider class before (and, for the happy path, after) ``from_env()``."""

    FRIENDLY_NAME = "Fake Provider"

    @classmethod
    def gating_env_vars(cls) -> tuple[str, ...]:
        return ("FAKE_PROVIDER_API_KEY",)

    @classmethod
    def help_summary(cls) -> str:
        return "a fake provider (test double)"

    @classmethod
    def display_label(cls) -> str:
        return cls.FRIENDLY_NAME

    @classmethod
    def provider_type(cls) -> ProviderType:
        # Arbitrary real member; only reached on the happy path, which none
        # of these tests exercise (both stand-ins below fail or return None).
        return ProviderType.CUSTOM


@pytest.fixture
def _isolated_startup(monkeypatch):
    """Run ``configure_providers()`` against a clean, single-provider roster.

    Mirrors ``tests/test_unconfigured_provider_reporting.py``'s
    ``unconfigured_providers`` fixture: clears the live
    :class:`ModelProviderRegistry` and resets the recorded #116 message, then
    restores both afterwards.
    """
    registry = ModelProviderRegistry()
    saved_providers = dict(registry._providers)
    saved_initialized = dict(registry._initialized_providers)
    registry._providers.clear()
    registry._initialized_providers.clear()

    monkeypatch.setattr(server, "_provider_configuration_error", None, raising=False)

    yield

    registry._providers.clear()
    registry._initialized_providers.clear()
    registry._providers.update(saved_providers)
    registry._initialized_providers.update(saved_initialized)


def test_malformed_registry_fails_startup_naming_the_file(tmp_path, monkeypatch, _isolated_startup):
    """A provider whose registry file is present but broken must abort startup.

    The failure message must name the offending file so the operator fixes
    the right thing, and the condition must NOT be reported through the #116
    "no providers configured" path -- that would tell the operator to set an
    API key that is already set.
    """
    bad_config = tmp_path / "fake_provider_models.json"
    bad_config.write_text(json.dumps({"models": "not-a-list"}))

    class _BrokenRegistryProvider(_FakeProviderBase):
        """``from_env()`` behaves as a real provider's would once
        ``providers/registries/base.py`` raises ``ModelRegistryConfigError``
        for a schema violation instead of swallowing it (see module docstring)."""

        @classmethod
        def from_env(cls):
            raw = json.loads(bad_config.read_text())
            assert not isinstance(raw.get("models"), list), "fixture must actually be malformed"
            raise ModelRegistryConfigError(str(bad_config), "malformed model registry")

    monkeypatch.setattr("providers.registry.REGISTERED_PROVIDER_CLASSES", [_BrokenRegistryProvider])

    with pytest.raises(ModelRegistryConfigError) as excinfo:
        server.configure_providers()

    assert str(bad_config) in str(excinfo.value)
    # Must fail loudly, not get recorded as "no providers configured".
    assert server.get_provider_configuration_error() is None


def test_merely_unset_api_key_still_starts_cleanly(monkeypatch, _isolated_startup):
    """Control: an absent API key is 'not configured', not 'configured but broken'.

    No registry file is involved -- ``from_env()`` returns ``None``, exactly
    as a real provider does when its gating env var is unset -- so this must
    go through the ordinary #116 path unchanged.
    """

    class _UnconfiguredProvider(_FakeProviderBase):
        """``from_env()`` returns ``None``, as a real provider does when its
        API key (here ``FAKE_PROVIDER_API_KEY``) is not set."""

        @classmethod
        def from_env(cls):
            return None

    monkeypatch.setattr("providers.registry.REGISTERED_PROVIDER_CLASSES", [_UnconfiguredProvider])

    server.configure_providers()  # must not raise

    message = server.get_provider_configuration_error()
    assert message is not None
    assert "At least one API configuration is required" in message
    assert "FAKE_PROVIDER_API_KEY" in message


# --------------------------------------------------------------------------
# Real registry / real provider coverage.
#
# The tests above pin the contract at the configure_providers() boundary
# with stand-ins, which is why the underlying fix could be dormant despite
# them passing (issue #130). These instead go through the real parse site
# (providers/registries/base.py) and, for the last one, a real provider
# (CustomProvider) and configure_providers() together.
# --------------------------------------------------------------------------


def test_real_registry_malformed_json_raises_naming_file(tmp_path):
    """A genuinely malformed ``tmp_path`` JSON file fails ``load_or_raise()``.

    Uses ``CustomEndpointModelRegistry`` (a real, unmodified registry class)
    with an explicit ``config_path`` so no stand-in is involved.
    """
    bad_config = tmp_path / "custom_models.json"
    bad_config.write_text("{ not valid json")

    with pytest.raises(ModelRegistryConfigError) as excinfo:
        CustomEndpointModelRegistry.load_or_raise(config_path=str(bad_config))

    assert str(bad_config) in str(excinfo.value)

    # The plain constructor's contract is unchanged: it still degrades to an
    # empty registry rather than raising (see providers/registries/base.py
    # and tests/test_openrouter_registry.py::test_invalid_json_config, which
    # pins this for the sibling OpenRouterModelRegistry).
    lenient = CustomEndpointModelRegistry(config_path=str(bad_config))
    assert lenient.list_models() == []


def test_real_registry_schema_violation_raises_naming_file(tmp_path):
    """A schema-invalid entry (legacy ``max_tokens`` field) is escalated too."""
    bad_config = tmp_path / "custom_models.json"
    bad_config.write_text(json.dumps({"models": [{"model_name": "test/model", "max_tokens": 1234}]}))

    with pytest.raises(ModelRegistryConfigError) as excinfo:
        CustomEndpointModelRegistry.load_or_raise(config_path=str(bad_config))

    assert str(bad_config) in str(excinfo.value)
    assert "max_output_tokens" in str(excinfo.value)

    # Unchanged contract for the plain constructor: still a plain ValueError
    # (see tests/test_openrouter_registry.py::test_backwards_compatibility_max_tokens
    # for the sibling registry pinning this).
    with pytest.raises(ValueError):
        CustomEndpointModelRegistry(config_path=str(bad_config))


def test_real_registry_absent_file_still_degrades_quietly(tmp_path):
    """Control: a genuinely absent file must NOT raise -- only a present-but-broken one does."""
    missing = tmp_path / "custom_models.json"
    assert not missing.exists()

    registry = CustomEndpointModelRegistry.load_or_raise(config_path=str(missing))

    assert registry.list_models() == []
    assert registry.config_error is None


def test_configure_providers_raises_for_real_provider_with_malformed_registry(tmp_path, monkeypatch, _isolated_startup):
    """End-to-end: ``configure_providers()`` with a real, broken ``CustomProvider`` registry.

    Unlike the stand-in-based tests earlier in this module, this goes
    through the actual provider (``CustomProvider.from_env()``) and the
    actual registry parse site (``providers/registries/base.py``), so it
    would have caught the fix being dormant.
    """
    bad_config = tmp_path / "custom_models.json"
    bad_config.write_text("{ not valid json")

    # CustomProvider._registry is a process-wide class-level cache populated
    # by __init__(); reset it so an earlier test's successful load (or this
    # one's) can't leak across tests and mask the failure being exercised
    # here -- this is the flakiness the module docstring warns about.
    monkeypatch.setattr(CustomProvider, "_registry", None)
    monkeypatch.setattr("providers.registry.REGISTERED_PROVIDER_CLASSES", [CustomProvider])
    monkeypatch.setenv("CUSTOM_API_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CUSTOM_MODELS_CONFIG_PATH", str(bad_config))

    with pytest.raises(ModelRegistryConfigError) as excinfo:
        server.configure_providers()

    assert str(bad_config) in str(excinfo.value)
    # Must fail loudly, not get recorded as "no providers configured".
    assert server.get_provider_configuration_error() is None
