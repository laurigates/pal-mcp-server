"""Tests for scripts/generate_model_entries.py (#149).

The generator writes config entries that nobody reads line by line once there
are hundreds of them, so three properties are pinned: every field comes from
the catalog record (an absent limit stays absent), no judgment fields are
written and every entry is disabled, and the text splice changes the file only
by appending.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from providers.registries.base import CAPABILITY_FIELD_NAMES

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_model_entries.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_model_entries", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_model_entries"] = module
    spec.loader.exec_module(module)
    return module


gen = _load_module()

OPENROUTER_FACTS = {
    "id": "vendor/model-x",
    "name": "Vendor: Model X",
    "context_window": 262_144,
    "max_output_tokens": None,
    "input_modalities": ["text", "image"],
    "supported_parameters": ["tools", "reasoning", "max_tokens"],
}

MODELSDEV_FACTS = {
    "id": "model-y",
    "name": "Model Y",
    "context_window": 400_000,
    "max_output_tokens": 128_000,
    "input_modalities": ["text"],
    "attachment": False,
    "tool_call": True,
    "reasoning": False,
    "temperature": True,
}


class TestEntryFromFacts:
    def test_openrouter_mapping(self):
        entry = gen.entry_from_facts(OPENROUTER_FACTS, "openrouter")
        assert entry == {
            "model_name": "vendor/model-x",
            "description": "Vendor: Model X",
            "context_window": 262_144,
            "supports_images": True,
            "supports_function_calling": True,
            "supports_extended_thinking": True,
            # temperature absent from supported_parameters -> rejected
            "supports_temperature": False,
            "temperature_constraint": "fixed",
            "enabled_by_default": False,
        }

    def test_modelsdev_mapping(self):
        entry = gen.entry_from_facts(MODELSDEV_FACTS, "modelsdev")
        assert entry == {
            "model_name": "model-y",
            "description": "Model Y",
            "context_window": 400_000,
            "max_output_tokens": 128_000,
            "supports_images": False,
            "supports_function_calling": True,
            "supports_extended_thinking": False,
            "enabled_by_default": False,
        }

    @pytest.mark.parametrize("facts,source", [(OPENROUTER_FACTS, "openrouter"), (MODELSDEV_FACTS, "modelsdev")])
    def test_never_writes_judgment_fields(self, facts, source):
        entry = gen.entry_from_facts(facts, source)
        assert "intelligence_score" not in entry
        assert "aliases" not in entry
        assert "allow_code_generation" not in entry
        assert set(entry) <= CAPABILITY_FIELD_NAMES

    def test_unknown_temperature_leaves_default(self):
        facts = {**MODELSDEV_FACTS, "temperature": None}
        assert "supports_temperature" not in gen.entry_from_facts(facts, "modelsdev")


class TestGenerateFor:
    def test_skips_models_dev_status_deprecated(self, tmp_path, monkeypatch):
        """#171: a model models.dev flags as deprecated is never generated."""
        (tmp_path / "gemini_models.json").write_text('{\n  "models": []\n}\n')
        monkeypatch.setattr(gen, "CONF_DIR", tmp_path)
        model = {"limit": {"context": 1000}, "modalities": {"input": ["text"], "output": ["text"]}}
        catalogs = {
            "modelsdev": {
                "google": {
                    "models": {
                        "gemini-served": {**model, "name": "Served"},
                        "gemini-gone": {**model, "name": "Gone", "status": "deprecated"},
                    }
                }
            }
        }
        target = gen.Target("gemini_models.json", "modelsdev", "google", "Gemini")
        entries, err = gen.generate_for(target, catalogs)
        assert err is None
        assert [e["model_name"] for e in entries] == ["gemini-served"]


class TestSpliceEntries:
    ENTRIES = [{"model_name": "a", "enabled_by_default": False}]

    def test_preserves_existing_formatting(self):
        text = '{\n  "_README": {"k": 1},\n  "models": [\n    {"model_name": "x", "levels": ["low", "high"]}\n  ]\n}\n'
        result = gen.splice_entries(text, self.ENTRIES)
        assert result.startswith(
            '{\n  "_README": {"k": 1},\n  "models": [\n    {"model_name": "x", "levels": ["low", "high"]},\n'
        )
        assert json.loads(result)["models"][-1] == self.ENTRIES[0]

    def test_empty_models_array(self):
        result = gen.splice_entries('{\n  "models": []\n}\n', self.ENTRIES)
        assert json.loads(result) == {"models": self.ENTRIES}

    def test_refuses_when_models_is_not_last(self):
        with pytest.raises(ValueError):
            gen.splice_entries('{"models": [], "other": 1}', self.ENTRIES)

    def test_no_entries_is_a_no_op(self):
        text = '{"models": []}'
        assert gen.splice_entries(text, []) is text

    def test_generated_config_loads_in_the_registry(self, tmp_path):
        from providers.registries.openrouter import OpenRouterModelRegistry

        entry = gen.entry_from_facts(OPENROUTER_FACTS, "openrouter")
        path = tmp_path / "openrouter_models.json"
        path.write_text(gen.splice_entries('{\n  "models": []\n}\n', [entry]))

        capabilities = OpenRouterModelRegistry(config_path=str(path)).resolve("vendor/model-x")
        assert capabilities is not None
        assert capabilities.enabled_by_default is False
        assert capabilities.supports_temperature is False
