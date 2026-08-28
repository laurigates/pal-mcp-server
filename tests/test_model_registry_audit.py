"""Tests for scripts/audit_model_registry.py.

Two things are pinned here. The first is the audit's own logic, exercised
against hand-built catalog fixtures so no test touches the network.

The second matters more: ``test_capability_fields_match_dataclass`` asserts the
script's ``CAPABILITY_FIELDS`` set equals the real ``ModelCapabilities``
dataclass. That set is a hand-maintained copy, and the whole point of the
audit's SCHEMA check is to catch config keys the registry drops silently. If
the copy drifts from the dataclass, the check starts reporting good fields as
unknown -- or worse, stops reporting bad ones -- and nothing else in the suite
would notice.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "audit_model_registry.py"


def _load_module():
    """Import the script by path -- scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("audit_model_registry", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_model_registry"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


@pytest.fixture
def catalogs():
    """Minimal stand-ins for the two live catalogs."""
    return {
        "openrouter": {
            "data": [
                {
                    "id": "vendor/live-model",
                    "canonical_slug": "vendor/live-model",
                    "name": "Live",
                    "context_length": 200_000,
                    "top_provider": {"max_completion_tokens": 64_000},
                    "created": 1_780_000_000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
                {
                    "id": "vendor/expiring-model",
                    "canonical_slug": "vendor/expiring-model",
                    "name": "Expiring",
                    "context_length": 100_000,
                    "top_provider": {"max_completion_tokens": 32_000},
                    "expiration_date": "2026-09-30",
                    "created": 1_770_000_000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
                {
                    "id": "vendor/brand-new",
                    "canonical_slug": "vendor/brand-new",
                    "name": "Brand New",
                    "context_length": 400_000,
                    "top_provider": {"max_completion_tokens": 128_000},
                    "created": 1_790_000_000,
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
            ]
        },
        "modelsdev": {
            "google": {
                "id": "google",
                "models": {
                    "gemini-test": {
                        "id": "gemini-test",
                        "name": "Gemini Test",
                        "limit": {"context": 1_000_000, "output": 65_536},
                        "release_date": "2026-01-01",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "reasoning": True,
                        "tool_call": True,
                        "attachment": True,
                    }
                },
            }
        },
    }


def _write_conf(tmp_path: Path, filename: str, models: list[dict]) -> Path:
    conf = tmp_path / "conf"
    conf.mkdir(exist_ok=True)
    (conf / filename).write_text(json.dumps({"_README": {}, "models": models}))
    return conf


@pytest.fixture
def conf_dir(tmp_path, monkeypatch):
    """Point the module's CONF_DIR at a temp dir for the duration of a test."""

    def _make(filename: str, models: list[dict]) -> Path:
        conf = _write_conf(tmp_path, filename, models)
        monkeypatch.setattr(audit, "CONF_DIR", conf)
        return conf

    return _make


def _kinds(findings, kind):
    return [f for f in findings if f.kind == kind]


class TestDeprecationDetection:
    def test_model_absent_from_catalog_is_deprecated(self, conf_dir, catalogs):
        conf_dir("openrouter_models.json", [{"model_name": "vendor/withdrawn", "aliases": ["gone"]}])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        deprecated = _kinds(findings, "deprecated")
        assert [f.model_name for f in deprecated] == ["vendor/withdrawn"]
        assert deprecated[0].extra["aliases"] == ["gone"]

    def test_expiration_date_is_reported(self, conf_dir, catalogs):
        conf_dir(
            "openrouter_models.json",
            [{"model_name": "vendor/expiring-model", "context_window": 100_000, "max_output_tokens": 32_000}],
        )
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        deprecated = _kinds(findings, "deprecated")
        assert len(deprecated) == 1
        assert "2026-09-30" in deprecated[0].detail

    def test_openrouter_absence_is_confirmed_modelsdev_absence_is_review(self, conf_dir, catalogs):
        """The authority difference between the two catalogs is load-bearing.

        models.dev is community-maintained and omits live models, so an absence
        there must never read as proof a provider withdrew something.
        """
        conf_dir("openrouter_models.json", [{"model_name": "vendor/withdrawn"}])
        or_target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        or_findings, _ = audit.audit_target(or_target, catalogs, top_n=0)
        assert _kinds(or_findings, "deprecated")[0].confidence == "confirmed"

        conf_dir("gemini_models.json", [{"model_name": "gemini-withdrawn"}])
        md_target = audit.Target("gemini_models.json", "modelsdev", "google", "Gemini")
        md_findings, _ = audit.audit_target(md_target, catalogs, top_n=0)
        assert _kinds(md_findings, "deprecated")[0].confidence == "review"

    def test_canonical_slug_and_variant_suffix_are_not_removals(self, catalogs):
        """A re-slugged or :batch-suffixed id is a rename, not a withdrawal."""
        catalogs["openrouter"]["data"].append(
            {
                "id": "vendor/new-slug",
                "canonical_slug": "vendor/old-slug",
                "context_length": 1000,
                "top_provider": {},
                "architecture": {"output_modalities": ["text"]},
            }
        )
        index = audit.openrouter_index(catalogs["openrouter"])
        assert "vendor/old-slug" in index
        assert "vendor/new-slug" in index


class TestStaleMetadata:
    def test_context_window_mismatch_is_reported(self, conf_dir, catalogs):
        conf_dir(
            "openrouter_models.json",
            [{"model_name": "vendor/live-model", "context_window": 128_000, "max_output_tokens": 64_000}],
        )
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        stale = _kinds(findings, "stale")
        assert len(stale) == 1
        assert stale[0].extra == {"field": "context_window", "configured": 128_000, "catalog": 200_000}

    def test_matching_metadata_produces_no_finding(self, conf_dir, catalogs):
        conf_dir(
            "openrouter_models.json",
            [{"model_name": "vendor/live-model", "context_window": 200_000, "max_output_tokens": 64_000}],
        )
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        assert _kinds(findings, "stale") == []


class TestCandidateAdditions:
    def test_unconfigured_live_model_is_a_candidate(self, conf_dir, catalogs):
        conf_dir("openrouter_models.json", [{"model_name": "vendor/live-model"}])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, meta = audit.audit_target(target, catalogs, top_n=10)
        names = {f.model_name for f in _kinds(findings, "missing")}
        assert "vendor/brand-new" in names
        assert "vendor/live-model" not in names
        assert meta["candidates_total"] >= 1

    def test_candidates_are_newest_first(self, conf_dir, catalogs):
        conf_dir("openrouter_models.json", [])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=10)
        assert [f.model_name for f in _kinds(findings, "missing")][0] == "vendor/brand-new"

    def test_top_n_zero_suppresses_candidates(self, conf_dir, catalogs):
        conf_dir("openrouter_models.json", [])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        assert _kinds(findings, "missing") == []

    @pytest.mark.parametrize(
        "model_id",
        ["text-embedding-3-large", "gpt-4o-tts", "imagen-4", "veo-3", "whisper-1", "llama-guard-3"],
    )
    def test_non_chat_models_are_not_candidates(self, model_id):
        assert not audit.is_chat_model({"id": model_id, "output_modalities": ["text"]})

    def test_audio_only_output_is_not_a_chat_model(self):
        assert not audit.is_chat_model({"id": "some-speech-model", "output_modalities": ["audio"]})

    @pytest.mark.parametrize("model_id", ["anthropic/claude-opus-4.8:batch", "z-ai/glm-5.3:free", "~z-ai/glm-latest"])
    def test_variant_and_wildcard_ids_are_not_candidates(self, model_id):
        assert not audit.is_candidate_id(model_id)

    def test_plain_id_is_a_candidate(self):
        assert audit.is_candidate_id("anthropic/claude-opus-4.8")


class TestAliasAndSchemaChecks:
    def test_duplicate_alias_in_one_file_is_a_collision(self):
        findings = audit.check_aliases(
            "x.json",
            [
                {"model_name": "model-a", "aliases": ["pro"]},
                {"model_name": "model-b", "aliases": ["PRO"]},
            ],
        )
        assert len(findings) == 1
        assert "already maps to model-a" in findings[0].detail

    def test_same_alias_on_same_model_is_not_a_collision(self):
        assert audit.check_aliases("x.json", [{"model_name": "model-a", "aliases": ["pro", "pro"]}]) == []

    def test_unknown_field_is_flagged(self):
        findings = audit.check_schema("x.json", [{"model_name": "m", "suports_images": True}])
        assert len(findings) == 1
        assert "suports_images" in findings[0].detail

    def test_known_fields_pass(self):
        assert audit.check_schema("x.json", [{"model_name": "m", "context_window": 1, "aliases": []}]) == []

    def test_missing_model_name_is_flagged(self):
        findings = audit.check_schema("x.json", [{"aliases": ["x"]}])
        assert any("no model_name" in f.detail for f in findings)


class TestUnverifiableTargets:
    def test_unverifiable_target_still_runs_local_checks(self, conf_dir, catalogs):
        """DIAL/custom/azure get no catalog, but alias and schema checks apply."""
        conf_dir(
            "dial_models.json",
            [
                {"model_name": "a", "aliases": ["dup"]},
                {"model_name": "b", "aliases": ["dup"], "bogus_field": 1},
            ],
        )
        target = audit.Target("dial_models.json", "unverifiable", label="DIAL", reason="no public catalog")
        findings, meta = audit.audit_target(target, catalogs, top_n=10)
        assert meta["catalog"] == 0
        assert meta["reason"] == "no public catalog"
        assert _kinds(findings, "alias_collision")
        assert _kinds(findings, "schema")
        # No catalog means no deprecation or candidate claims can be made.
        assert _kinds(findings, "deprecated") == []
        assert _kinds(findings, "missing") == []


class TestRealConfigs:
    """Guards against the audit silently auditing nothing."""

    @pytest.mark.parametrize("target", audit.TARGETS, ids=lambda t: t.filename)
    def test_every_target_config_exists_and_parses(self, target):
        models, err = audit.read_config(audit.CONF_DIR / target.filename)
        assert err is None, err
        assert isinstance(models, list)

    def test_shipped_configs_have_no_alias_collisions(self):
        for target in audit.TARGETS:
            models, _ = audit.read_config(audit.CONF_DIR / target.filename)
            assert audit.check_aliases(target.filename, models) == []

    def test_shipped_configs_use_only_capability_fields(self):
        for target in audit.TARGETS:
            models, _ = audit.read_config(audit.CONF_DIR / target.filename)
            assert audit.check_schema(target.filename, models) == []


def test_capability_fields_match_dataclass():
    """The script's field list is a copy; drift makes the SCHEMA check lie.

    A field added to ModelCapabilities but not here would be reported as an
    unknown field in every config that uses it. A field removed there but left
    here would stop being reported even though the registry now drops it.
    """
    from providers.shared.model_capabilities import ModelCapabilities

    actual = {f.name for f in dataclasses.fields(ModelCapabilities)}
    assert audit.CAPABILITY_FIELDS == actual, (
        "scripts/audit_model_registry.py CAPABILITY_FIELDS is out of step with "
        f"ModelCapabilities: only in script {audit.CAPABILITY_FIELDS - actual}, "
        f"only in dataclass {actual - audit.CAPABILITY_FIELDS}"
    )
