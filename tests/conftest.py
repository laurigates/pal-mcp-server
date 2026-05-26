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

# Provider classes are needed by the session bootstrap fixture below.
from providers.gemini import GeminiModelProvider  # noqa: E402
from providers.openai import OpenAIModelProvider  # noqa: E402
from providers.registry import ModelProviderRegistry  # noqa: E402
from providers.shared import ProviderType  # noqa: E402
from providers.xai import XAIModelProvider  # noqa: E402


def _set_dummy_keys_if_missing():
    """Set dummy API keys only when they are completely absent."""
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        if not os.environ.get(var):
            os.environ[var] = "dummy-key-for-tests"


def _register_default_providers():
    """Register the providers every unit test expects to be available."""
    ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)

    # Register CUSTOM provider when running prompt regression integration tests.
    if os.getenv("CUSTOM_API_URL") and "test_prompt_regression.py" in os.getenv("PYTEST_CURRENT_TEST", ""):
        from providers.custom import CustomProvider

        def custom_provider_factory(api_key=None):
            base_url = os.getenv("CUSTOM_API_URL", "")
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)


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
def _ensure_default_providers_registered():
    """Restore the default provider registry before each test.

    Several tests intentionally clear the registry singleton (or call
    ``unregister_provider`` in ``setup_method`` / ``teardown_method``).
    Without re-registration here, subsequent tests that rely on
    Gemini/OpenAI/XAI being present fail in test-order-dependent ways.

    This fixture runs unconditionally (autouse) — it deals with
    *registry* state and is orthogonal to the ``mock_provider`` fixture,
    which only stubs ``BaseTool.is_effective_auto_mode``.
    """
    registry = ModelProviderRegistry()
    if ProviderType.GOOGLE not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    if ProviderType.OPENAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    if ProviderType.XAI not in registry._providers:
        ModelProviderRegistry.register_provider(ProviderType.XAI, XAIModelProvider)

    if (
        os.getenv("CUSTOM_API_URL")
        and "test_prompt_regression.py" in os.getenv("PYTEST_CURRENT_TEST", "")
        and ProviderType.CUSTOM not in registry._providers
    ):
        from providers.custom import CustomProvider

        def custom_provider_factory(api_key=None):
            base_url = os.getenv("CUSTOM_API_URL", "")
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)


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
def _clear_model_restriction_env(monkeypatch):
    """Ensure per-test isolation from user-defined model restriction env vars."""

    restriction_vars = [
        "OPENAI_ALLOWED_MODELS",
        "GOOGLE_ALLOWED_MODELS",
        "XAI_ALLOWED_MODELS",
        "OPENROUTER_ALLOWED_MODELS",
        "DIAL_ALLOWED_MODELS",
    ]

    for var in restriction_vars:
        monkeypatch.delenv(var, raising=False)


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
