# PAL MCP Server task runner.
#
# The quality gate itself lives in code_quality_checks.sh (also called by CI);
# these recipes wrap it rather than re-implementing the steps, so local and CI
# run the same commands.

_default:
    @just --list

# Full quality gate: lockfile check, sync, lint, format, type check, unit tests
check:
    ./code_quality_checks.sh

# Lint and auto-fix
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Type check (ty is pre-1.0; advisory)
typecheck:
    uv run ty check .

# Unit tests (no API keys needed)
test:
    uv run pytest tests/ -m "not integration"

# Integration tests (needs Ollama + CUSTOM_API_URL)
test-integration:
    ./run_integration_tests.sh

# --- model registry upkeep -------------------------------------------------
#
# Mechanical drift detection against two public, keyless catalogs:
# OpenRouter /api/v1/models (authoritative, carries expiration_date) and
# models.dev/api.json (community-maintained, covers google/openai/xai/opencode).
#
# The script reports; it never edits conf/*.json. Turning a candidate into a
# config entry is judgment — see .claude/skills/model-registry-audit/SKILL.md.

# Audit conf/*_models.json for deprecated, stale, and missing models
models-audit *ARGS:
    uv run python scripts/audit_model_registry.py --cache-dir .cache/model-catalogs {{ARGS}}

# Audit one provider, e.g. `just models-audit-one openrouter_models.json`
models-audit-one FILE *ARGS:
    uv run python scripts/audit_model_registry.py --cache-dir .cache/model-catalogs --only {{FILE}} {{ARGS}}

# Machine-readable audit (what the scheduled workflow consumes)
models-audit-json:
    @uv run python scripts/audit_model_registry.py --cache-dir .cache/model-catalogs --json

# Re-audit from the cached catalogs without hitting the network
models-audit-offline *ARGS:
    uv run python scripts/audit_model_registry.py --cache-dir .cache/model-catalogs --offline {{ARGS}}

# CI gate: exit 1 when deprecated/stale/collision/schema drift exists
models-audit-strict:
    uv run python scripts/audit_model_registry.py --cache-dir .cache/model-catalogs --fail-on-drift
