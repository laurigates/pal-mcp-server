"""Tests for the simulator's provider-agnostic CI selection (issue #132).

``communication_simulator_test.py`` could not be run by CI for two reasons, both
pinned here:

* every scenario hard-coded ``"model": "flash"``, so a checkout whose only
  provider is a local Ollama got "Model 'flash' is not available" before any
  request reached the wire; and
* the run's result was seeded from the whole test registry, so a subset run
  counted its passes against ~35 entries and reported FAILURE however the tests
  went -- ``--quick`` could not go green.

These run in the normal unit suite: nothing here starts a server or needs a key.
"""

import ast
from pathlib import Path

import pytest

from communication_simulator_test import CommunicationSimulator
from simulator_tests import TEST_REGISTRY
from simulator_tests.base_test import DEFAULT_SIMULATOR_MODEL, get_simulator_model

REPO_ROOT = Path(__file__).resolve().parent.parent


def _model_literals(source_path: Path) -> list[str]:
    """Every string a ``"model"`` key is given literally in this module."""
    tree = ast.parse(source_path.read_text())
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "model" and isinstance(value, ast.Constant):
                literals.append(value.value)
    return literals


class TestSimulatorModel:
    def test_defaults_to_flash(self, monkeypatch):
        monkeypatch.delenv("SIMULATOR_MODEL", raising=False)
        assert get_simulator_model() == DEFAULT_SIMULATOR_MODEL == "flash"

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("SIMULATOR_MODEL", "ci-local")
        assert get_simulator_model() == "ci-local"

    def test_empty_value_is_not_a_model_name(self, monkeypatch):
        """An unset-but-exported variable must not ask for the model ''."""
        monkeypatch.setenv("SIMULATOR_MODEL", "")
        assert get_simulator_model() == DEFAULT_SIMULATOR_MODEL


class TestProviderAgnosticScenarios:
    def test_registry_marks_some_scenarios_agnostic(self):
        assert [name for name, cls in TEST_REGISTRY.items() if cls.provider_agnostic]

    @pytest.mark.parametrize("name", sorted(name for name, cls in TEST_REGISTRY.items() if cls.provider_agnostic))
    def test_agnostic_scenarios_name_no_model_literally(self, name):
        """A provider-agnostic scenario asks for SIMULATOR_MODEL, never a name.

        This is the regression guard: a scenario that reintroduces
        ``"model": "flash"`` is provider-specific again and would fail CI's
        Ollama-only run with a model-availability error.
        """
        module = Path(TEST_REGISTRY[name].__module__.replace(".", "/") + ".py")
        assert _model_literals(REPO_ROOT / module) == []

    def test_scenarios_that_pin_a_provider_are_not_marked(self):
        """O3 and Gemini thinking-config routing are the point of those tests."""
        for name in ("o3_model_selection", "model_thinking_config"):
            assert not TEST_REGISTRY[name].provider_agnostic


class TestCIMode:
    def test_ci_mode_selects_exactly_the_agnostic_scenarios(self):
        simulator = CommunicationSimulator(ci_mode=True)
        expected = sorted(name for name, cls in TEST_REGISTRY.items() if cls.provider_agnostic)
        assert sorted(simulator.selected_tests) == expected

    def test_results_cover_only_the_selected_scenarios(self):
        """Seeding results from the whole registry made every subset run red."""
        simulator = CommunicationSimulator(ci_mode=True)
        assert set(simulator.test_results) == set(simulator.selected_tests)

    def test_a_clean_subset_run_reports_success(self):
        """The summary is the exit code's source; it must be able to say SUCCESS."""
        simulator = CommunicationSimulator(ci_mode=True)
        for name in simulator.test_results:
            simulator.test_results[name] = True
        assert simulator.print_test_summary() is True

    def test_full_run_still_covers_every_scenario(self):
        simulator = CommunicationSimulator()
        assert set(simulator.test_results) == set(TEST_REGISTRY)
