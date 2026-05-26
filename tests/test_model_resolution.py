"""
Unit tests for ``utils.model_resolution.resolve_model_for_context``.

These tests pin the fallback cascade so the refactor that extracted
this helper out of ``server.reconstruct_thread_context`` (issue #6)
does not silently regress the ordering: existing-context →
requested-model → tool-category-fallback → first-globally-available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.models import ToolModelCategory
from utils.model_context import ModelContext
from utils.model_resolution import resolve_model_for_context


@pytest.fixture
def mock_tool():
    """Stand-in tool with a stable ``get_model_category`` return."""
    tool = MagicMock()
    tool.get_model_category.return_value = ToolModelCategory.BALANCED
    return tool


def _stub_registry(monkeypatch, *, providers=None, preferred=None, available=None):
    """
    Patch ``ModelProviderRegistry`` lookups used inside
    ``resolve_model_for_context``. Returns nothing — the patches stay
    active for the lifetime of the calling test.

    Args:
        providers: dict mapping model_name → provider object (or None).
            ``get_provider_for_model`` returns ``providers[name]`` when
            provided, else ``MagicMock()`` (i.e. "provider available").
        preferred: dict mapping category → model name for the
            tool-category fallback. ``KeyError`` ⇒ raise ``ValueError``.
        available: list of globally-available model names.
    """
    from providers import registry

    def fake_get_provider(name):
        if providers is None:
            return MagicMock()
        return providers.get(name)

    def fake_get_preferred(category):
        if preferred is None:
            raise ValueError("no preferred")
        return preferred[category]

    def fake_get_available(*args, **kwargs):
        return list(available or [])

    monkeypatch.setattr(registry.ModelProviderRegistry, "get_provider_for_model", staticmethod(fake_get_provider))
    monkeypatch.setattr(
        registry.ModelProviderRegistry,
        "get_preferred_fallback_model",
        classmethod(lambda cls, c: fake_get_preferred(c)),
    )
    monkeypatch.setattr(
        registry.ModelProviderRegistry,
        "get_available_model_names",
        classmethod(lambda cls, provider_type=None: fake_get_available(provider_type)),
    )


class TestResolveModelForContext:
    """Verify the cascade order documented in model_resolution.py."""

    def test_returns_existing_context_unchanged(self, mock_tool, monkeypatch):
        """Step 1: an explicit ModelContext is trusted verbatim."""
        _stub_registry(monkeypatch)
        existing = ModelContext("preset-model")

        result = resolve_model_for_context(
            tool=mock_tool,
            requested_model_name="ignored-name",
            existing_model_context=existing,
        )

        assert result is existing  # identity, not just equality

    def test_uses_requested_model_when_available(self, mock_tool, monkeypatch):
        """Step 2: the requested model resolves to a ModelContext when its provider is available."""
        _stub_registry(monkeypatch, providers={"gemini-pro": MagicMock()})

        result = resolve_model_for_context(
            tool=mock_tool,
            requested_model_name="gemini-pro",
            existing_model_context=None,
        )

        assert isinstance(result, ModelContext)
        assert result.model_name == "gemini-pro"

    def test_falls_back_to_tool_category_when_modelcontext_raises(self, mock_tool, monkeypatch):
        """Step 3: ModelContext.from_arguments ValueError → tool-category fallback."""
        _stub_registry(
            monkeypatch,
            providers={"fallback-model": MagicMock()},
            preferred={ToolModelCategory.BALANCED: "fallback-model"},
        )

        with patch.object(ModelContext, "from_arguments", side_effect=ValueError("no model")):
            result = resolve_model_for_context(
                tool=mock_tool,
                requested_model_name="bogus",
                existing_model_context=None,
            )

        assert isinstance(result, ModelContext)
        assert result.model_name == "fallback-model"
        mock_tool.get_model_category.assert_called()

    def test_falls_back_to_first_available_when_no_tool(self, monkeypatch):
        """Step 4: with no tool, the registry's first available model is used."""
        _stub_registry(
            monkeypatch,
            providers={"first-available": MagicMock()},
            available=["first-available", "second"],
        )

        with patch.object(ModelContext, "from_arguments", side_effect=ValueError("no model")):
            result = resolve_model_for_context(
                tool=None,
                requested_model_name="bogus",
                existing_model_context=None,
            )

        assert isinstance(result, ModelContext)
        assert result.model_name == "first-available"

    def test_raises_when_nothing_is_available(self, mock_tool, monkeypatch):
        """No requested model resolves and no fallback exists → ValueError propagates."""
        _stub_registry(monkeypatch, providers={}, available=[])

        with patch.object(ModelContext, "from_arguments", side_effect=ValueError("no model")):
            with pytest.raises(ValueError):
                resolve_model_for_context(
                    tool=mock_tool,
                    requested_model_name="bogus",
                    existing_model_context=None,
                )

    def test_swaps_when_resolved_model_has_no_provider(self, mock_tool, monkeypatch):
        """Step 5: ModelContext resolves, but provider missing → swap to fallback."""
        _stub_registry(
            monkeypatch,
            providers={"primary": None, "secondary": MagicMock()},
            preferred={ToolModelCategory.BALANCED: "secondary"},
        )

        result = resolve_model_for_context(
            tool=mock_tool,
            requested_model_name="primary",
            existing_model_context=None,
        )

        assert isinstance(result, ModelContext)
        assert result.model_name == "secondary"

    def test_provider_missing_and_no_fallback_raises(self, mock_tool, monkeypatch):
        """Provider missing + no fallback → user-facing 'not available' ValueError."""
        _stub_registry(monkeypatch, providers={"primary": None}, available=[])

        with pytest.raises(ValueError, match="not available"):
            resolve_model_for_context(
                tool=mock_tool,
                requested_model_name="primary",
                existing_model_context=None,
            )

    def test_no_requested_model_uses_default(self, mock_tool, monkeypatch):
        """No requested_model_name → ModelContext.from_arguments({}) uses DEFAULT_MODEL."""
        _stub_registry(monkeypatch, providers={"default-model": MagicMock()})

        with patch.object(ModelContext, "from_arguments", return_value=ModelContext("default-model")) as mock_from_args:
            result = resolve_model_for_context(
                tool=mock_tool,
                requested_model_name=None,
                existing_model_context=None,
            )

        # The empty-arguments path is what the orchestrator uses when
        # the client did not pin a model.
        mock_from_args.assert_called_once_with({})
        assert result.model_name == "default-model"
