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

    # OpenRouter's top_provider.max_completion_tokens follows whichever upstream
    # provider ranks first that week (#234), so only a configured value above it
    # -- the direction that can fail a request -- is drift.
    def test_openrouter_max_output_below_catalog_is_not_stale(self, conf_dir, catalogs):
        conf_dir(
            "openrouter_models.json",
            [{"model_name": "vendor/live-model", "context_window": 200_000, "max_output_tokens": 32_000}],
        )
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        assert _kinds(findings, "stale") == []

    def test_openrouter_max_output_above_catalog_is_stale(self, conf_dir, catalogs):
        conf_dir(
            "openrouter_models.json",
            [{"model_name": "vendor/live-model", "context_window": 200_000, "max_output_tokens": 128_000}],
        )
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        stale = _kinds(findings, "stale")
        assert len(stale) == 1
        assert stale[0].extra == {"field": "max_output_tokens", "configured": 128_000, "catalog": 64_000}

    def test_modelsdev_max_output_below_catalog_is_still_stale(self, conf_dir, catalogs):
        conf_dir(
            "gemini_models.json",
            [{"model_name": "gemini-test", "context_window": 1_000_000, "max_output_tokens": 32_768}],
        )
        target = audit.Target("gemini_models.json", "modelsdev", "google", "Gemini")
        findings, _ = audit.audit_target(target, catalogs, top_n=0)
        stale = _kinds(findings, "stale")
        assert len(stale) == 1
        assert stale[0].extra == {"field": "max_output_tokens", "configured": 32_768, "catalog": 65_536}


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

    def test_disabled_entry_is_not_a_candidate(self, conf_dir, catalogs):
        """#149: a model triaged in switched off is configured, not MISSING."""
        conf_dir("openrouter_models.json", [{"model_name": "vendor/brand-new", "enabled_by_default": False}])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=10)
        assert "vendor/brand-new" not in {f.model_name for f in _kinds(findings, "missing")}
        assert _kinds(findings, "schema") == []

    def test_id_claimed_by_an_alias_is_not_a_candidate(self, conf_dir, catalogs):
        """An entry named after an existing alias would override what that name resolves to."""
        conf_dir("openrouter_models.json", [{"model_name": "vendor/live-model", "aliases": ["Vendor/Brand-New"]}])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=10)
        assert "vendor/brand-new" not in {f.model_name for f in _kinds(findings, "missing")}

    def test_expiring_model_is_not_a_candidate(self, conf_dir, catalogs):
        """Adding it would only trade a MISSING finding for a DEPRECATED one."""
        conf_dir("openrouter_models.json", [])
        target = audit.Target("openrouter_models.json", "openrouter", label="OpenRouter")
        findings, _ = audit.audit_target(target, catalogs, top_n=10)
        assert "vendor/expiring-model" not in {f.model_name for f in _kinds(findings, "missing")}

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


class TestModelsDevStatusDeprecated:
    """#171: models.dev marks withdrawn models ``"status": "deprecated"``.

    The flag is not proof of withdrawal (some flagged models still serve), so a
    configured one is a ``[review]`` finding, and an unconfigured one is never
    proposed as an addition.
    """

    TARGET = audit.Target("gemini_models.json", "modelsdev", "google", "Gemini")

    @pytest.fixture
    def flagged(self, catalogs):
        catalogs["modelsdev"]["google"]["models"]["gemini-flagged"] = {
            "id": "gemini-flagged",
            "name": "Gemini Flagged",
            "limit": {"context": 1_000_000, "output": 65_536},
            "release_date": "2026-06-01",
            "modalities": {"input": ["text"], "output": ["text"]},
            "status": "deprecated",
        }
        return catalogs

    def test_deprecated_status_is_not_a_candidate(self, conf_dir, flagged):
        conf_dir("gemini_models.json", [])
        findings, meta = audit.audit_target(self.TARGET, flagged, top_n=10)
        names = {f.model_name for f in _kinds(findings, "missing")}
        assert "gemini-flagged" not in names
        assert "gemini-test" in names
        assert meta["candidates_total"] == 1

    def test_configured_deprecated_status_is_a_review_finding(self, conf_dir, flagged):
        conf_dir(
            "gemini_models.json",
            [{"model_name": "gemini-flagged", "aliases": ["flag"], "context_window": 1_000_000}],
        )
        findings, _ = audit.audit_target(self.TARGET, flagged, top_n=0)
        deprecated = _kinds(findings, "deprecated")
        assert [f.model_name for f in deprecated] == ["gemini-flagged"]
        assert deprecated[0].detail == "catalog sets status=deprecated"
        assert deprecated[0].confidence == "review"
        assert deprecated[0].extra["aliases"] == ["flag"]

    def test_other_status_values_are_live(self, conf_dir, flagged):
        flagged["modelsdev"]["google"]["models"]["gemini-flagged"]["status"] = "beta"
        conf_dir("gemini_models.json", [{"model_name": "gemini-flagged"}])
        findings, _ = audit.audit_target(self.TARGET, flagged, top_n=10)
        assert _kinds(findings, "deprecated") == []
        assert "gemini-flagged" not in {f.model_name for f in _kinds(findings, "missing")}


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


class TestProviderReferenceCheck:
    """The orphan-reference check, pinned in both directions.

    A provider module hardcodes canonical ids (PRIMARY_MODEL, FALLBACK_MODEL,
    per-category preference lists) outside the registry. When a model leaves the
    config those strings keep pointing at it and category routing resolves to
    nothing -- silently, because the tests exercise the resolver rather than the
    constants. These tests exist so the check that catches it cannot rot into a
    no-op.
    """

    def _module(self, tmp_path, monkeypatch, source: str) -> str:
        providers = tmp_path / "providers"
        providers.mkdir(exist_ok=True)
        (providers / "fake.py").write_text(source)
        monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
        return "fake.py"

    def test_hardcoded_id_absent_from_config_is_flagged(self, tmp_path, monkeypatch):
        mod = self._module(tmp_path, monkeypatch, 'PRIMARY_MODEL = "gpt-4"\nFALLBACK_MODEL = "gpt-5.2"\n')
        findings = audit.check_provider_references("openai_models.json", mod, [{"model_name": "gpt-5.2"}])
        assert [f.model_name for f in findings] == ["gpt-4"]
        assert findings[0].kind == "orphan_ref"

    def test_configured_id_is_not_flagged(self, tmp_path, monkeypatch):
        mod = self._module(tmp_path, monkeypatch, 'PRIMARY_MODEL = "gpt-5.2"\n')
        assert audit.check_provider_references("openai_models.json", mod, [{"model_name": "gpt-5.2"}]) == []

    def test_an_alias_counts_as_configured(self, tmp_path, monkeypatch):
        """Providers legitimately pin an alias rather than a canonical name."""
        mod = self._module(tmp_path, monkeypatch, 'PRIMARY_MODEL = "gpt5.2"\n')
        models = [{"model_name": "gpt-5.2", "aliases": ["gpt5.2"]}]
        assert audit.check_provider_references("openai_models.json", mod, models) == []

    @pytest.mark.parametrize(
        "literal",
        [
            "https://api.openai.com/v1",
            "application/json",
            "OPENAI_MODELS_CONFIG_PATH",
            "text",
            "provider",
            # These pass MODEL_ID_SHAPE outright -- lowercase, separated, with a
            # digit -- and are excluded only by NON_MODEL_LITERALS.
            "utf-8",
            "iso-8859-1",
            "sha-256",
            "http/1.1",
        ],
    )
    def test_non_model_literals_are_not_flagged(self, tmp_path, monkeypatch, literal):
        """The shape filter runs over every string in the module, so it must not
        mistake URLs, MIME types, encodings or env-var names for model ids."""
        mod = self._module(tmp_path, monkeypatch, f"VALUE = {literal!r}\n")
        assert audit.check_provider_references("x.json", mod, [{"model_name": "m-1"}]) == []

    def test_missing_or_unparseable_module_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
        assert audit.check_provider_references("x.json", "nonexistent.py", []) == []
        (tmp_path / "providers").mkdir()
        (tmp_path / "providers" / "broken.py").write_text("def (:\n")
        assert audit.check_provider_references("x.json", "broken.py", []) == []

    def test_shipped_providers_have_no_orphan_references(self):
        """The live gate: every id hardcoded in providers/ resolves in its config."""
        for target in audit.TARGETS:
            if not target.module:
                continue
            models, err = audit.read_config(audit.CONF_DIR / target.filename)
            assert err is None, err
            orphans = audit.check_provider_references(target.filename, target.module, models)
            assert orphans == [], f"{target.module}: {[f.model_name for f in orphans]}"


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


class TestOpenCodeGoGateway:
    """#235: with ``OPENCODE_API_KEY`` set, the gateway's own ``/v1/models`` is the authority.

    On 2026-11-16 models.dev's ``opencode-go`` slice listed ``mimo-v2.5`` and
    ``mimo-v2.5-pro`` as live while the gateway answered 410 for both, so the
    keyless audit kept proposing them as additions.
    """

    TARGET = next(t for t in audit.TARGETS if t.filename == "opencode_go_models.json")

    GATEWAY_PAYLOAD = {
        "object": "list",
        "data": [
            {
                "id": "mimo-v2.6-flash",
                "name": "MiMo V2.6 Flash",
                "type": "chat",
                "release_date": "2026-11-01",
                "modalities": {"input": ["text", "image/png"], "output": ["text"]},
                "capabilities": {"tools": {"supported": True}, "reasoning": {"supported": True}, "temperature": True},
                "limit": {"context": 1_000_000, "output": 128_000},
            },
            {
                "id": "glm-5.3",
                "name": "GLM 5.3",
                "type": "chat",
                "release_date": "2026-09-01",
                "modalities": {"input": ["text"], "output": ["text"]},
                "capabilities": {"tools": {"supported": True}, "reasoning": {"supported": False}},
                "limit": {"context": 200_000, "output": 64_000},
            },
        ],
    }

    @pytest.fixture
    def go_catalogs(self):
        model = {"limit": {"context": 1000}, "modalities": {"input": ["text"], "output": ["text"]}}
        return {
            "modelsdev": {
                "opencode-go": {
                    "models": {
                        "glm-5.3": {**model, "name": "GLM 5.3", "release_date": "2026-09-01"},
                        "mimo-v2.5": {**model, "name": "MiMo V2.5", "release_date": "2026-10-01"},
                    }
                }
            },
            "opencode_go": self.GATEWAY_PAYLOAD,
        }

    def test_target_names_the_gateway(self):
        assert self.TARGET.gateway == "opencode_go"

    def test_gateway_index_normalises_the_payload(self):
        index = audit.gateway_index(self.GATEWAY_PAYLOAD)
        facts = index["mimo-v2.6-flash"]
        assert facts["context_window"] == 1_000_000
        assert facts["max_output_tokens"] == 128_000
        assert facts["input_modalities"] == ["text", "image"]
        assert facts["tool_call"] is True
        assert facts["reasoning"] is True
        assert facts["temperature"] is True
        assert index["glm-5.3"]["temperature"] is None

    def test_gateway_absence_is_a_confirmed_deprecation(self, conf_dir, go_catalogs):
        conf_dir("opencode_go_models.json", [{"model_name": "glm-5.3"}, {"model_name": "mimo-v2.5"}])
        findings, meta = audit.audit_target(self.TARGET, go_catalogs, top_n=10)
        deprecated = _kinds(findings, "deprecated")
        assert [(f.model_name, f.confidence) for f in deprecated] == [("mimo-v2.5", "confirmed")]
        assert "opencode_go" in deprecated[0].detail
        assert meta["source"] == "opencode_go"
        assert meta["catalog"] == 2

    def test_candidates_come_from_the_gateway_only(self, conf_dir, go_catalogs):
        conf_dir("opencode_go_models.json", [{"model_name": "glm-5.3"}])
        findings, _ = audit.audit_target(self.TARGET, go_catalogs, top_n=10)
        missing = _kinds(findings, "missing")
        assert [f.model_name for f in missing] == ["mimo-v2.6-flash"]
        assert missing[0].extra["context_window"] == 1_000_000

    def test_without_the_gateway_models_dev_stays_review(self, conf_dir, go_catalogs):
        del go_catalogs["opencode_go"]
        conf_dir("opencode_go_models.json", [{"model_name": "glm-5.3"}, {"model_name": "gone"}])
        findings, meta = audit.audit_target(self.TARGET, go_catalogs, top_n=10)
        assert [(f.model_name, f.confidence) for f in _kinds(findings, "deprecated")] == [("gone", "review")]
        assert meta["source"] == "modelsdev"
        assert {f.model_name for f in _kinds(findings, "missing")} == {"mimo-v2.5"}


class TestLoadGatewayCatalogs:
    """The gateway is fetched only with a key, and never fails the keyless run."""

    def test_no_key_fetches_nothing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)

        def _boom(*_a, **_k):
            raise AssertionError("fetched without a key")

        monkeypatch.setattr(audit, "fetch_json", _boom)
        assert audit.load_gateway_catalogs(tmp_path, offline=False) == {}

    def test_key_fetches_with_bearer_and_caches(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        calls = []

        def _fetch(url, headers=None, timeout=45):
            calls.append((url, headers))
            return {"data": [{"id": "glm-5.3"}]}

        monkeypatch.setattr(audit, "fetch_json", _fetch)
        out = audit.load_gateway_catalogs(tmp_path, offline=False)
        assert out == {"opencode_go": {"data": [{"id": "glm-5.3"}]}}
        assert calls == [("https://opencode.ai/zen/go/v1/models", {"Authorization": "Bearer test-key"})]
        cache_text = (tmp_path / "opencode_go.json").read_text()
        assert json.loads(cache_text) == {"data": [{"id": "glm-5.3"}]}
        assert "test-key" not in cache_text

    def test_fetch_failure_falls_back_to_models_dev(self, monkeypatch, capsys):
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")

        def _fail(*_a, **_k):
            raise audit.urllib.error.URLError("down")

        monkeypatch.setattr(audit, "fetch_json", _fail)
        assert audit.load_gateway_catalogs(None, offline=False) == {}
        assert "opencode_go" in capsys.readouterr().err

    def test_offline_reads_the_cache(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
        (tmp_path / "opencode_go.json").write_text(json.dumps({"data": [{"id": "glm-5.3"}]}))
        assert audit.load_gateway_catalogs(tmp_path, offline=True) == {"opencode_go": {"data": [{"id": "glm-5.3"}]}}
        assert audit.load_gateway_catalogs(tmp_path / "empty", offline=True) == {}
