# PRD: Dynamic model registry

| | |
|---|---|
| Status | Approved |
| Issue | [#240](https://github.com/laurigates/pal-mcp-server/issues/240) |
| Decision record | [ADR-0001](../adr/0001-dynamic-model-registry.md) |
| Date | 2026-11-17 |

## Problem

PAL describes every model it can call in `conf/*_models.json`. Those files change only when someone edits them, and the providers' catalogs change every week. Between edits the registry drifts:

- **Removed models stay callable by name** and fail at request time. On 2026-11-16, `openrouter_models.json` held 26 slugs OpenRouter no longer serves, and `opencode_go_models.json` held 14 ids the gateway rejects with 400 or 410 (fixed in #236).
- **Wrong limits produce bad requests.** PAL sends `max_output_tokens` as `max_tokens` on every call (#129). On 2026-11-16, 43 OpenRouter entries had a context window or output limit that no longer matched the catalog. OpenRouter's `max_completion_tokens` for a slug changes from week to week, so no static value stays correct (#234).
- **New models are invisible** until someone runs `just models-generate` and merges the result.

The weekly audit (`.github/workflows/model-registry-audit.yml`) finds the drift but does not fix it, and its PR depends on a person acting on it. When the 2026-11-09 run failed to push (#233), a week of drift accumulated with no signal beyond a red workflow run.

## Goal

The facts PAL sends to providers reflect the provider's catalog as of the last successful fetch, so drift lasts until the next server start instead of until the next manual refresh. Curated judgment (which models auto mode prefers, which short names resolve where) stays reviewed and versioned.

## Non-goals

- **Automating judgment.** `intelligence_score`, aliases, `allow_code_generation` and `enabled_by_default` stay hand-curated. A model the catalog adds appears disabled, as `just models-generate` already does.
- **Using models.dev at runtime.** It is community-maintained and was wrong four ways on 2026-11-16 (#235, the Gemini findings in #236). It stays an input to the weekly audit only.
- **Custom.** It has no public catalog; their names are chosen by the operator.
- **Changing tool schemas.** The auto-mode `model` field lists only the top-ranked models (`tools/shared/base_tool.py`, `get_model_field_schema`), and new catalog models are disabled, so a refresh must not change any tool description.

## Users and scenarios

| Scenario | Expected behaviour |
|---|---|
| A user calls an OpenRouter model whose output limit changed upstream | PAL sends the current limit. No 400 from an over-limit `max_tokens`. |
| A model is withdrawn from OpenRouter | After the next refresh it is no longer offered or accepted by name. The error names the model as unavailable instead of surfacing a provider 404. |
| A provider releases a model | It is callable by its full id after the next refresh, without a PAL release. It stays out of auto mode until someone curates it. |
| PAL starts offline, or the catalog endpoint is down | PAL starts with the last cached catalog, or with the bundled JSON if no cache exists, and logs which one. Startup does not wait on the network. |
| CI and unit tests | Registry content is deterministic: the bundled JSON only, no network. |
| An operator pins a registry file (`OPENROUTER_MODELS_CONFIG_PATH` etc.) | The pinned file is used as-is and nothing is fetched for that provider. |

## Requirements

### Functional

1. **Two layers.** The effective entry for a model is its catalog facts with the curated overlay applied on top. The overlay wins field by field.
2. **Per-provider sources.** Each provider declares an authoritative source, or none (see ADR-0001 § Sources). A provider with no authoritative source keeps today's static file.
3. **Non-blocking startup.** Registries load synchronously from the cache, or the bundled JSON when there is no cache. The network refresh runs in the background, and the registry swaps to the refreshed data when it arrives.
4. **Refresh on every startup, cache as fallback.** Every start triggers a background fetch. Fetched catalogs are written under the state directory, which `utils/state_dir.py :: get_state_dir` already resolves from `PAL_STATE_DIR` and XDG. The cache exists only to serve the next start until its own fetch lands, and to cover offline starts. A failed refresh keeps the previous cache and logs a warning. A long-running server refreshes only at startup; there is no periodic refresh in this design.
5. **Validation before use.** A fetched record that fails schema validation is dropped, not partially applied. Numeric limits are bounded to sane ranges before they reach a request.
6. **Removal only on an authoritative absence.** A model leaves the effective registry only when an authoritative source omits it. A curated entry whose model disappears is disabled, not deleted, and the next audit reports it.
7. **On by default, with an off switch.** Fetching is enabled unless `PAL_DYNAMIC_REGISTRY=0` is set. The test suite sets it to 0.
8. **Observability.** `listmodels` (or `version`) shows, for each provider, which source supplied the registry (live, cache with its age, or bundled) and when it was last refreshed.

### Non-functional

- Startup latency stays unchanged: no network call on the import path.
- A cold fetch adds no more than one request per provider at startup. Measured on 2026-11-16: OpenRouter about 0.07 s for 0.9 MB.
- No new runtime dependency. `httpx` is already used by the providers, and the audit script uses `urllib`.

## Success measures

- After phase 1, the weekly audit reports zero confirmed `stale` and `deprecated` findings for `openrouter_models.json` beyond those the curated overlay deliberately pins.
- No provider 400 caused by an over-limit `max_tokens` on OpenRouter. There is no baseline count for this today; phase 1 should add logging that makes it countable.
- Unit-test runtime and determinism are unchanged: no network calls under pytest.

## Phases

| Phase | Provider | Source | Depends on |
|---|---|---|---|
| 1 | OpenRouter | `GET https://openrouter.ai/api/v1/models` (keyless) | — |
| 2 | Gemini | `models.list` (needs `GEMINI_API_KEY`) | Phase 1 machinery |
| 3 | OpenAI | `/v1/models` `shutdown_date` only. It carries no limits, so limits stay curated | Phase 1 machinery |
| 4 | OpenCode Go | Gateway `/v1/models` id list (needs `OPENCODE_API_KEY`) | #235 |

## Decisions

Lauri answered the draft's open questions on 2026-11-17:

| Question | Decision |
|---|---|
| Default on or off | **On by default.** `PAL_DYNAMIC_REGISTRY=0` turns it off (requirement 7) |
| Cache time-to-live | **Refresh in the background on every startup.** The cache is only a fallback until that fetch lands, and for offline starts (requirement 4) |
| Withdrawn curated models | **Disable them.** They leave auto mode, calls by name fail with a clear "unavailable" error, and the audit reports them (requirement 6) |
| Weekly audit output | **An issue, not a PR.** Once facts are dynamic, only curation work remains, and that needs no PR |
