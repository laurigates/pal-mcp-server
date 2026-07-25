"""
Pytest configuration for PAL MCP Server tests.

Fixture conventions
-------------------

* ``mock_provider`` is an **explicit** fixture (not autouse). It mocks
  ``BaseTool.is_effective_auto_mode`` to return ``False`` so tools with a
  configured ``DEFAULT_MODEL`` don't fall through to "auto mode requires
  model selection" errors. The fixture is applied automatically to every
  test that does **not** carry the ``@pytest.mark.no_mock_provider``
  marker — this preserves the original autouse-friendly UX without the
  text-matching on test names that used to live here.

* Tests that exercise real auto-mode / provider-resolution logic opt out
  with ``@pytest.mark.no_mock_provider`` (applied at function, class, or
  module level via ``pytestmark``).

* Session-scoped global state (the test workspace root, ``DEFAULT_MODEL``,
  registered providers, dummy API keys) is seeded once in
  :func:`pytest_configure` — not at module import time. The autouse
  ``_runtime_env`` fixture re-applies the per-test env defaults so
  individual tests that mutate ``DEFAULT_MODEL`` / etc. get a clean
  slate, and ``_ensure_default_providers_registered`` restores any
  providers a previous test unregistered.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# On macOS, the default pytest temp dir is typically under /var (e.g. /private/var/folders/...).
# If /var is considered a dangerous system path, tests must use a safe temp root (like /tmp).
if sys.platform == "darwin":
    os.environ["TMPDIR"] = "/tmp"
    # tempfile caches the temp dir after first lookup; clear it so pytest fixtures pick up TMPDIR.
    tempfile.tempdir = None

# Ensure the parent directory is in the Python path for imports.
# This must happen before any ``import config`` / ``import providers.*``.
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Widen the PAL workspace root for the test session so that existing tests
# which point ``working_directory_absolute_path`` at arbitrary ``tmp_path``
# values (typically ``/private/var/folders/...`` on macOS or ``/tmp/...``)
# continue to pass the workspace-containment check introduced for
# issues #4 / #5. Tests that specifically exercise the containment logic
# override ``PAL_WORKSPACE_ROOT`` via ``monkeypatch`` to tighten the root.
os.environ.setdefault("PAL_WORKSPACE_ROOT", "/")

import utils.env as env_config  # noqa: E402

# Ensure tests operate with runtime environment rather than .env overrides during imports.
# (The per-test ``_runtime_env`` fixture below re-applies this on every test.)
env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# The provider roster is needed by the session bootstrap fixture below.
from providers.registry import (  # noqa: E402
    PROVIDER_CLASS_BY_TYPE,
    REGISTERED_PROVIDER_CLASSES,
    ModelProviderRegistry,
)
from providers.shared import ProviderType  # noqa: E402

#: The provider set the bulk of the unit suite assumes is available.
DEFAULT_TEST_PROVIDERS: tuple[ProviderType, ...] = (
    ProviderType.GOOGLE,
    ProviderType.OPENAI,
    ProviderType.XAI,
)


def _set_dummy_keys_if_missing():
    """Set dummy API keys only when they are completely absent."""
    for provider_type in DEFAULT_TEST_PROVIDERS:
        env_var = PROVIDER_CLASS_BY_TYPE[provider_type].API_KEY_ENV
        if env_var and not os.environ.get(env_var):
            os.environ[env_var] = "dummy-key-for-tests"


def _register_default_providers():
    """Register the providers every unit test expects to be available."""
    for provider_type in DEFAULT_TEST_PROVIDERS:
        ModelProviderRegistry.register_provider(provider_type, PROVIDER_CLASS_BY_TYPE[provider_type])


def pytest_configure(config):
    """Register custom markers and run one-time session setup.

    ``pytest_configure`` fires *before* test modules are collected/imported,
    so this is the right place to seed ``DEFAULT_MODEL`` (some test
    modules do ``from config import DEFAULT_MODEL`` at import time and
    capture the value then). The per-test ``_runtime_env`` fixture
    re-applies the same defaults during execution so individual tests
    that mutate them get a clean slate.
    """
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line(
        "markers",
        "no_mock_provider: opt out of the default mock_provider fixture for tests "
        "that exercise real provider resolution / auto-mode logic.",
    )
    config.addinivalue_line(
        "markers",
        "ambient_provider_env: keep the real provider environment instead of the scrubbed one.",
    )

    # Seed the test-time defaults *before* collection so module-level
    # ``from config import DEFAULT_MODEL`` imports see the right value.
    os.environ.setdefault("DEFAULT_MODEL", "gemini-2.5-flash")

    # Force reload of config so any earlier-imported modules pick up the env var.
    import importlib

    import config as _config

    importlib.reload(_config)

    # One-time test environment bootstrap (replaces module-import side effects).
    _set_dummy_keys_if_missing()
    _register_default_providers()


def pytest_collection_modifyitems(session, config, items):
    """Apply the ``mock_provider`` fixture to every test that doesn't opt out.

    This replaces the old text-matching on test names. Tests that carry
    ``@pytest.mark.no_mock_provider`` (at any scope) keep their existing
    behaviour; everything else has ``mock_provider`` injected so the
    auto-mode short-circuit lands without each test having to ``request``
    the fixture explicitly.

    ``mock_provider`` is prepended (not appended) to ``fixturenames`` so
    it runs *before* ``setup_method`` — matching the historical autouse
    ordering that several test classes (e.g. ``TestModelProviderRegistry``)
    depend on for registry teardown semantics.
    """
    for item in items:
        if item.get_closest_marker("no_mock_provider") is None:
            if "mock_provider" not in item.fixturenames:
                # Prepend so it resolves before xunit ``setup_method`` —
                # autouse fixtures from this conftest will still come
                # first (they remain at the head of the list).
                item.fixturenames.insert(0, "mock_provider")


@pytest.fixture
def project_path(tmp_path):
    """
    Provides a temporary directory for tests.
    This ensures all file operations during tests are isolated.
    """
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def _isolated_provider_registry(request, monkeypatch):
    """Derived provider isolation: env + registry, for every test.

    Scrubs ``credential_env_vars()`` for every registered provider. Despite the
    name that is wider than API keys: it covers ``REQUIRED_ENV``,
    ``OPTIONAL_ENV`` and the allow-lists, so ``CUSTOM_API_URL``,
    ``AZURE_OPENAI_ENDPOINT``, every ``*_MODELS_CONFIG_PATH`` and every
    ``*_ALLOWED_MODELS`` are unset inside a test unless the test sets them
    itself. That breadth is deliberate -- it is what stops a developer's .env
    from reaching the assertions (issue #66) -- but it means a test that
    expects to inherit any of those from the ambient environment must set it in
    the test body or opt out with ``@pytest.mark.ambient_provider_env``.
    """
    import utils.model_restrictions as model_restrictions

    keep_ambient_env = (
        request.node.get_closest_marker("ambient_provider_env") is not None
        or request.node.get_closest_marker("integration") is not None
    )

    if not keep_ambient_env:
        for provider_cls in REGISTERED_PROVIDER_CLASSES:
            for var in provider_cls.credential_env_vars():
                monkeypatch.delenv(var, raising=False)

    registry = ModelProviderRegistry()
    saved_providers = dict(registry._providers)
    saved_initialized = dict(registry._initialized_providers)
    registry._providers.clear()
    registry._initialized_providers.clear()

    if keep_ambient_env:
        # Ambient-env tests (the local-Ollama integration suite) want whatever
        # the real environment configures. Derived from the roster, so this
        # replaces the old hardcoded CUSTOM_API_URL/PYTEST_CURRENT_TEST branch.
        for provider_cls in REGISTERED_PROVIDER_CLASSES:
            if provider_cls.is_configured():
                ModelProviderRegistry.register_provider(provider_cls.provider_type(), provider_cls)
    else:
        for provider_type in DEFAULT_TEST_PROVIDERS:
            provider_cls = PROVIDER_CLASS_BY_TYPE[provider_type]
            if provider_cls.API_KEY_ENV:
                monkeypatch.setenv(provider_cls.API_KEY_ENV, "dummy-key-for-tests")
            ModelProviderRegistry.register_provider(provider_type, provider_cls)

    model_restrictions._restriction_service = None
    try:
        yield
    finally:
        model_restrictions._restriction_service = None
        registry._providers.clear()
        registry._initialized_providers.clear()
        registry._providers.update(saved_providers)
        registry._initialized_providers.update(saved_initialized)


@pytest.fixture
def mock_provider(monkeypatch):
    """Disable ``BaseTool.is_effective_auto_mode`` for tests that don't need it.

    Most tests run with a fixed ``DEFAULT_MODEL`` (set by
    ``_runtime_env``) and expect tools to treat that model as
    available. This fixture stubs ``is_effective_auto_mode`` to always
    return ``False`` so tools skip the "auto mode requires explicit
    model" branch.

    Applied automatically via :func:`pytest_collection_modifyitems` to
    every test that does not carry ``@pytest.mark.no_mock_provider``.
    Tests that exercise the real auto-mode logic opt out with that
    marker (at function, class, or module level).
    """
    from tools.shared.base_tool import BaseTool

    monkeypatch.setattr(BaseTool, "is_effective_auto_mode", lambda self: False)


@pytest.fixture(autouse=True)
def _runtime_env(monkeypatch):
    """Default tests to runtime environment visibility with a stable DEFAULT_MODEL.

    This replaces the old module-import-time ``os.environ["DEFAULT_MODEL"] = ...``
    + ``importlib.reload(config)`` side effects. Per-test scoping is
    necessary because individual tests mutate ``DEFAULT_MODEL`` /
    ``PAL_MCP_FORCE_ENV_OVERRIDE`` and we want them reset on entry.
    """
    monkeypatch.setenv("PAL_MCP_FORCE_ENV_OVERRIDE", "false")
    env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setenv("DEFAULT_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("MAX_CONVERSATION_TURNS", "50")

    import importlib
    import sys

    import config
    import utils.conversation_memory as conversation_memory

    importlib.reload(config)
    importlib.reload(conversation_memory)

    test_conversation_module = sys.modules.get("tests.test_conversation_memory")
    if test_conversation_module is not None:
        test_conversation_module.MAX_CONVERSATION_TURNS = conversation_memory.MAX_CONVERSATION_TURNS

    try:
        yield
    finally:
        env_config.reload_env()
