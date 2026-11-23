"""Dynamic OpenRouter registry built from the provider catalog (#240, ADR-0001 phase 1).

Pinned behaviour:
  * the merge -- catalog facts win for limits and capability flags, the
    bundled entry keeps every curated field, a curated model the catalog drops
    is disabled (not deleted), a new catalog model arrives disabled, and a
    malformed or out-of-range catalog value never reaches an entry;
  * the cache -- atomic write, unreadable cache ignored;
  * the registry -- reads the cache when on, the bundled file when
    ``PAL_DYNAMIC_REGISTRY=0`` or a file is pinned, and a refresh reloads
    live registries in place while a failed one keeps the previous data;
  * parity with the stdlib copies in scripts/ that the CI audit imports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from providers.registries import catalog
from providers.registries import openrouter as openrouter_registry_module
from providers.registries.openrouter import (
    OpenRouterModelRegistry,
    refresh_openrouter_catalog,
    start_openrouter_catalog_refresh,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED = json.loads((REPO_ROOT / "conf" / "openrouter_models.json").read_text())["models"]


def _record(model_id: str, *, context=200_000, output=32_000, params=("tools", "temperature"), **extra) -> dict:
    record = {
        "id": model_id,
        "name": f"Name of {model_id}",
        "context_length": context,
        "top_provider": {"context_length": context, "max_completion_tokens": output},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": list(params),
    }
    record.update(extra)
    return record


def _curated(name: str, **fields) -> dict:
    entry = {
        "model_name": name,
        "aliases": [f"{name.split('/')[-1]}-alias"],
        "context_window": 100_000,
        "max_output_tokens": 8_000,
        "supports_images": False,
        "supports_function_calling": False,
        "supports_extended_thinking": False,
        "intelligence_score": 17,
        "allow_code_generation": True,
        "description": "curated description",
    }
    entry.update(fields)
    return entry


@pytest.fixture
def dynamic_on(monkeypatch):
    monkeypatch.setenv("PAL_DYNAMIC_REGISTRY", "1")
    monkeypatch.delenv("OPENROUTER_MODELS_CONFIG_PATH", raising=False)
    openrouter_registry_module._reset_catalog_state()
    yield
    openrouter_registry_module._reset_catalog_state()


def _bundled_catalog(drop: str | None = None, output: int = 4_321) -> dict:
    """A catalog listing every bundled OpenRouter model, with one optionally withdrawn."""
    records = [
        _record(m["model_name"], context=777_000, output=output)
        for m in BUNDLED
        if m["model_name"] != drop and not m.get("provider")
    ]
    return {"data": records}


def _aliased_enabled_entry() -> dict:
    return next(m for m in BUNDLED if m.get("aliases") and m.get("enabled_by_default", True) and not m.get("provider"))


class TestMerge:
    def test_catalog_facts_override_and_curated_fields_survive(self):
        curated = [_curated("vendor/model-a")]
        records = [
            _record(
                "vendor/model-a",
                context=400_000,
                output=64_000,
                params=("tools", "reasoning"),
                architecture={"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            )
        ]

        merged, summary = catalog.merge_openrouter_catalog(curated, records)

        entry = merged[0]
        assert entry["context_window"] == 400_000
        assert entry["max_output_tokens"] == 64_000
        assert entry["supports_images"] is True
        assert entry["supports_function_calling"] is True
        assert entry["supports_extended_thinking"] is True
        for curated_field in ("aliases", "intelligence_score", "allow_code_generation", "description"):
            assert entry[curated_field] == curated[0][curated_field]
        assert summary == {"withdrawn": [], "added": []}
        assert curated[0]["context_window"] == 100_000, "the bundled list must not be mutated"

    def test_withdrawn_curated_model_is_disabled_not_deleted(self):
        curated = [_curated("vendor/gone"), _curated("vendor/kept")]

        merged, summary = catalog.merge_openrouter_catalog(curated, [_record("vendor/kept")])

        names = [m["model_name"] for m in merged]
        assert "vendor/gone" in names
        gone = merged[names.index("vendor/gone")]
        assert gone["enabled_by_default"] is False
        assert gone["context_window"] == 100_000, "a withdrawn entry keeps its curated facts"
        assert summary["withdrawn"] == ["vendor/gone"]

    def test_new_catalog_model_arrives_disabled_without_judgment_fields(self):
        merged, summary = catalog.merge_openrouter_catalog(
            [_curated("vendor/old")], [_record("vendor/old"), _record("vendor/brand-new", params=("tools",))]
        )

        new = next(m for m in merged if m["model_name"] == "vendor/brand-new")
        assert new["enabled_by_default"] is False
        assert "aliases" not in new and "intelligence_score" not in new
        assert new["context_window"] == 200_000
        assert new["supports_temperature"] is False, "a model whose params omit temperature rejects it"
        assert summary["added"] == ["vendor/brand-new"]

    @pytest.mark.parametrize(
        "model_id, extra",
        [
            ("vendor/model:free", {}),
            ("~vendor/router-latest", {}),
            ("vendor/text-embedding-9", {}),
            ("vendor/expiring", {"expiration_date": "2026-12-01"}),
            ("vendor/painter", {"architecture": {"input_modalities": ["text"], "output_modalities": ["image"]}}),
        ],
    )
    def test_non_candidates_are_not_added(self, model_id, extra):
        merged, summary = catalog.merge_openrouter_catalog([], [_record(model_id, **extra)])
        assert merged == [] and summary["added"] == []

    def test_free_variant_and_canonical_slug_count_as_present(self):
        curated = [_curated("vendor/base"), _curated("vendor/renamed")]
        records = [_record("vendor/base:free"), _record("vendor/new-slug", canonical_slug="vendor/renamed")]

        merged, summary = catalog.merge_openrouter_catalog(curated, records)

        assert summary["withdrawn"] == []
        assert all(
            m.get("enabled_by_default", True) for m in merged if m["model_name"] in {"vendor/base", "vendor/renamed"}
        )
        assert "vendor/new-slug" not in summary["added"], "a re-slugged curated model is not a new model"

    def test_entry_routed_to_another_provider_passes_through(self):
        custom = _curated("local-llama", provider="custom")
        merged, summary = catalog.merge_openrouter_catalog([custom], [_record("vendor/other")])
        assert merged[0] == custom
        assert summary["withdrawn"] == []

    @pytest.mark.parametrize(
        "context, output, expect_context, expect_output",
        [
            (0, 8_000, 100_000, 8_000),  # zero context: keep curated, output still bounded by MAX
            (10**12, 8_000, 100_000, 8_000),  # absurd context: keep curated
            (200_000, 300_000, 200_000, 8_000),  # output above context: keep curated output
            ("200000", None, 100_000, 8_000),  # wrong types: keep curated
            (True, -5, 100_000, 8_000),
        ],
    )
    def test_out_of_range_limits_keep_the_curated_value(self, context, output, expect_context, expect_output):
        record = _record("vendor/model-a", context=context, output=output)
        merged, _ = catalog.merge_openrouter_catalog([_curated("vendor/model-a")], [record])
        assert merged[0]["context_window"] == expect_context
        assert merged[0]["max_output_tokens"] == expect_output

    @pytest.mark.parametrize(
        "payload", [None, [], {"data": "x"}, {"data": []}, {"data": [1, {"id": ""}, {"name": "x"}]}]
    )
    def test_payload_without_valid_records_is_rejected(self, payload):
        with pytest.raises(ValueError):
            catalog.validate_openrouter_catalog(payload)

    def test_malformed_records_are_dropped_not_applied(self):
        payload = {"data": [_record("vendor/good"), {"id": "vendor/bad", "top_provider": "nope"}, "junk"]}
        assert [r["id"] for r in catalog.validate_openrouter_catalog(payload)] == ["vendor/good"]


class TestSwitch:
    @pytest.mark.parametrize(
        "value, expected", [(None, True), ("1", True), ("true", True), ("0", False), ("off", False), ("False", False)]
    )
    def test_dynamic_registry_switch(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("PAL_DYNAMIC_REGISTRY", raising=False)
        else:
            monkeypatch.setenv("PAL_DYNAMIC_REGISTRY", value)
        assert catalog.dynamic_registry_enabled() is expected

    def test_suite_runs_with_the_dynamic_registry_off(self):
        assert catalog.dynamic_registry_enabled() is False
        assert start_openrouter_catalog_refresh() is None


class TestCache:
    def test_round_trip(self):
        catalog.write_cache("x.json", {"data": [1]}, 1234.5)
        assert catalog.read_cache("x.json") == ({"data": [1]}, 1234.5)
        leftovers = [p.name for p in catalog.cache_path("x.json").parent.iterdir() if p.name != "x.json"]
        assert leftovers == [], "the temp file must be renamed into place"

    def test_absent_and_corrupt_cache_read_as_none(self):
        assert catalog.read_cache("absent.json") is None
        path = catalog.cache_path("bad.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert catalog.read_cache("bad.json") is None
        path.write_text(json.dumps({"payload": {}}))
        assert catalog.read_cache("bad.json") is None


class TestRegistry:
    def test_cached_catalog_supplies_facts_and_disables_withdrawn(self, dynamic_on):
        withdrawn = _aliased_enabled_entry()
        catalog.write_cache("openrouter_models.json", _bundled_catalog(drop=withdrawn["model_name"]), 1000.0)

        registry = OpenRouterModelRegistry()

        assert registry.registry_source == ("cache", 1000.0)
        kept = next(m for m in BUNDLED if m["model_name"] != withdrawn["model_name"] and not m.get("provider"))
        assert registry.get_model_config(kept["model_name"]).max_output_tokens == 4_321
        gone = registry.get_model_config(withdrawn["model_name"])
        assert gone is not None and gone.enabled_by_default is False
        assert registry.resolve(withdrawn["aliases"][0]) is None, "a disabled entry answers to its canonical name only"

    def test_without_a_cache_the_bundled_file_is_used(self, dynamic_on):
        registry = OpenRouterModelRegistry()
        assert registry.registry_source == ("bundled", None)
        assert len(registry.list_models()) == len(BUNDLED)

    def test_switch_off_ignores_the_cache(self, dynamic_on, monkeypatch):
        catalog.write_cache("openrouter_models.json", _bundled_catalog(), 1000.0)
        monkeypatch.setenv("PAL_DYNAMIC_REGISTRY", "0")
        registry = OpenRouterModelRegistry()
        entry = BUNDLED[0]
        assert registry.get_model_config(entry["model_name"]).max_output_tokens == entry.get(
            "max_output_tokens", 32_768
        )
        assert "PAL_DYNAMIC_REGISTRY=0" in registry.describe_source()

    def test_pinned_file_ignores_the_cache(self, dynamic_on, tmp_path):
        catalog.write_cache("openrouter_models.json", _bundled_catalog(), 1000.0)
        pinned = tmp_path / "pinned.json"
        pinned.write_text(json.dumps({"models": [_curated("vendor/pinned")]}))
        registry = OpenRouterModelRegistry(config_path=str(pinned))
        assert registry.list_models() == ["vendor/pinned"]
        assert registry.registry_source == ("pinned", None)

    def test_invalid_cache_falls_back_to_bundled(self, dynamic_on):
        catalog.write_cache("openrouter_models.json", {"data": []}, 1000.0)
        registry = OpenRouterModelRegistry()
        assert registry.registry_source == ("bundled", None)
        assert len(registry.list_models()) == len(BUNDLED)

    def test_refresh_reloads_live_registries_and_writes_the_cache(self, dynamic_on, monkeypatch):
        registry = OpenRouterModelRegistry()
        monkeypatch.setattr(catalog, "fetch_json", lambda url, timeout=None: _bundled_catalog(output=1_111))

        assert refresh_openrouter_catalog() is True

        name = next(m["model_name"] for m in BUNDLED if not m.get("provider"))
        assert registry.get_model_config(name).max_output_tokens == 1_111
        assert registry.registry_source[0] == "live"
        assert "live OpenRouter catalog" in registry.describe_source()
        payload, _ = catalog.read_cache("openrouter_models.json")
        assert payload["data"][0]["top_provider"]["max_completion_tokens"] == 1_111

    def test_failed_refresh_keeps_previous_data(self, dynamic_on, monkeypatch, caplog):
        catalog.write_cache("openrouter_models.json", _bundled_catalog(output=2_222), 1000.0)
        registry = OpenRouterModelRegistry()

        def boom(url, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr(catalog, "fetch_json", boom)
        assert refresh_openrouter_catalog() is False

        name = next(m["model_name"] for m in BUNDLED if not m.get("provider"))
        assert registry.get_model_config(name).max_output_tokens == 2_222
        assert catalog.read_cache("openrouter_models.json")[1] == 1000.0
        assert "refresh failed" in caplog.text

    def test_invalid_fetched_payload_is_not_cached(self, dynamic_on, monkeypatch):
        catalog.write_cache("openrouter_models.json", _bundled_catalog(output=2_222), 1000.0)
        monkeypatch.setattr(catalog, "fetch_json", lambda url, timeout=None: {"error": "rate limited"})
        assert refresh_openrouter_catalog() is False
        assert catalog.read_cache("openrouter_models.json")[1] == 1000.0

    def test_unwritable_cache_still_applies_the_fetch(self, dynamic_on, monkeypatch):
        registry = OpenRouterModelRegistry()
        monkeypatch.setattr(catalog, "fetch_json", lambda url, timeout=None: _bundled_catalog(output=3_333))

        def no_write(*args, **kwargs):
            raise PermissionError("read-only home")

        monkeypatch.setattr(catalog, "write_cache", no_write)
        assert refresh_openrouter_catalog() is True
        name = next(m["model_name"] for m in BUNDLED if not m.get("provider"))
        assert registry.get_model_config(name).max_output_tokens == 3_333

    def test_background_refresh_runs_on_a_daemon_thread(self, dynamic_on, monkeypatch):
        calls = []
        monkeypatch.setattr(openrouter_registry_module, "refresh_openrouter_catalog", lambda: calls.append(1))
        thread = start_openrouter_catalog_refresh()
        assert thread is not None and thread.daemon
        thread.join(timeout=5)
        assert calls == [1]

    def test_pinned_env_var_turns_the_refresh_off(self, dynamic_on, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(tmp_path / "x.json"))
        assert start_openrouter_catalog_refresh() is None


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestParityWithScripts:
    """The CI audit runs without project deps, so scripts/ keeps stdlib copies of this mapping."""

    RECORDS = [
        _record("vendor/a", params=("tools", "reasoning", "temperature")),
        _record("vendor/b:free", canonical_slug="vendor/b-2026", params=()),
        _record(
            "vendor/c",
            output=None,
            architecture={"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            params=("include_reasoning",),
        ),
    ]

    def test_index_matches_audit_script(self):
        audit = _load_script("audit_model_registry")
        script_index = audit.openrouter_index({"data": self.RECORDS})
        runtime_index = catalog.openrouter_index(self.RECORDS)
        assert {k.lower(): v for k, v in script_index.items()} == runtime_index

    def test_new_entry_matches_generator_script(self):
        gen = _load_script("generate_model_entries")
        for facts in catalog.openrouter_index(self.RECORDS).values():
            assert catalog.entry_from_facts(facts) == gen.entry_from_facts(facts, "openrouter")

    def test_non_chat_markers_match_audit_script(self):
        audit = _load_script("audit_model_registry")
        assert catalog.NON_CHAT_MARKERS == audit.NON_CHAT_MARKERS
