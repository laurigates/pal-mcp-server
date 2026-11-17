# ADR-0001: Build model facts from provider catalogs at startup, with a curated overlay

| | |
|---|---|
| Status | Accepted (2026-11-17) |
| Date | 2026-11-17 |
| Issue | [#240](https://github.com/laurigates/pal-mcp-server/issues/240) |
| Requirements | [PRD: Dynamic model registry](../prd/dynamic-model-registry.md) |

## Context

Each provider's registry (`providers/registries/*.py`) subclasses `CustomModelRegistryBase`. It reads one JSON file from `conf/` (or from the path in an env var such as `OPENROUTER_MODELS_CONFIG_PATH`), and `reload()` rebuilds its name and alias maps from that file. Several providers build their registry at import time, for example `OpenAIModelProvider._ensure_registry()` at the bottom of `providers/openai.py`.

An entry mixes two kinds of field:

| Kind | Fields | Where the value comes from |
|---|---|---|
| Catalog facts | `context_window`, `max_output_tokens`, `supports_images`, `supports_function_calling`, `supports_extended_thinking`, `supports_temperature` | The provider's catalog. `scripts/generate_model_entries.py :: entry_from_facts` already maps catalog records to these fields |
| Curation | `aliases`, `intelligence_score`, `allow_code_generation`, `enabled_by_default`, `use_openai_response_api`, a hand-written `description`, non-default `temperature_constraint` | Human judgment. No catalog provides it |

In `openrouter_models.json`, 39 of 341 entries carry an `intelligence_score`. The rest contain only catalog facts plus `enabled_by_default`.

The 2026-11-16 refresh (#236, #238) showed how reliable each candidate source is:

| Source | Covers | Reliability observed |
|---|---|---|
| OpenRouter `/api/v1/models` | OpenRouter slugs | Authoritative: it is the serving list. `top_provider.max_completion_tokens` changes from week to week (#234), which is a reason to read it live, not a defect |
| Gemini `models.list` | Gemini direct | Returns `inputTokenLimit`, `outputTokenLimit`, `thinking`, `maxTemperature` (Gemini API reference). Image and tool support are not returned. Not yet exercised with a key |
| OpenAI `/v1/models` | OpenAI direct | Returns `id`, `created`, `owned_by`, `shutdown_date`, and no token limits (developers.openai.com API reference) |
| OpenCode Go gateway `/v1/models` | OpenCode Go | Authoritative for the curated id list. Delisted ids have kept routing for a while (`tests/test_opencode_go_provider.py`), so a missing id is a strong but not conclusive signal |
| models.dev | Gemini, OpenAI, xAI, OpenCode Go | Wrong on 2026-11-16 for `gemini-4-argon`'s output limit (32,768 against Google's documented 65,536), both deep-research context windows (131,072 against 1,048,576), and two OpenCode Go ids the gateway answers with 410 |
| xAI list endpoint | xAI direct | Fields not verified. The fetched API reference pages document only the Responses endpoints |

## Decision

1. **Split each entry into a facts layer and a curated overlay.** The facts layer is built from the provider's authoritative catalog. The overlay is a checked-in file holding only curation fields, keyed by `model_name`. The effective entry is the facts with the overlay applied field by field, so the overlay wins. An overlay entry may also pin a fact (for example a deliberately lower `max_output_tokens`); a pin is visible in review because the overlay contains nothing else.

2. **Fetch only from authoritative sources, never from models.dev.** Each registry declares its source, or none:

   | Provider | Facts source | Removal on absence |
   |---|---|---|
   | OpenRouter | Catalog (keyless) | Yes |
   | Gemini | `models.list` with key | Yes, for ids that `models.list` returned before |
   | OpenAI | `/v1/models` for `shutdown_date` only. Limits stay in the overlay | No. Disable after `shutdown_date` passes |
   | OpenCode Go | Gateway `/v1/models` ids only | No. Disable, and report to the audit (#235) |
   | xAI, DIAL, Azure, Custom | None. Static file as today | — |

   models.dev stays an input to the weekly audit.

3. **Load from the cache first and refresh in the background on every startup.** At import, a registry loads the cached facts, or the bundled JSON if there is no cache, exactly as fast as today. Every start then launches a background task that fetches the catalog, validates it, writes the cache atomically (temp file plus rename), and calls `reload()`. There is no time-to-live and no periodic refresh while the server runs. A failed fetch leaves the previous data in place and logs a warning. No request path waits on the network.

4. **Keep the bundled JSON as the floor.** `conf/*_models.json` remains in the package. It is the offline fallback, the test fixture, and the snapshot that `just models-generate` and the weekly audit keep reasonably fresh. A file pinned via a `*_MODELS_CONFIG_PATH` env var disables fetching for that provider.

5. **New catalog models arrive disabled.** A fact with no overlay entry gets `enabled_by_default: false`, the rule `entry_from_facts` already applies. It is callable by its full id but never chosen by auto mode, and it does not appear in the auto-mode `model` field description, which lists only top-ranked models.

6. **On by default, off under tests.** Fetching is enabled unless `PAL_DYNAMIC_REGISTRY=0` is set; `tests/conftest.py` sets it. The simulator's stub tier keeps using the bundled files.

7. **A withdrawn curated model is disabled, not deleted.** When an authoritative source drops a model that has an overlay entry, the effective entry gets `enabled_by_default: false`, and a call by name fails with an "unavailable" error that names the model. The overlay entry stays until a person removes it.

8. **The weekly audit opens an issue instead of a PR.** With facts coming from catalogs, its remaining output is curation work. That needs no committed report, and an issue avoids the branch-push path that failed in #233.

## Consequences

**Positive**
- OpenRouter limits are current as of the last server start, which resolves #234 without changing how the audit compares values.
- A withdrawn OpenRouter model stops being offered without a PAL release.
- The weekly audit shrinks to curation: models to score, aliases to move, overlay pins that disagree with the catalog.

**Negative**
- **Registry contents vary between runs and machines.** A bug report has to state the registry source and age. Requirement 8 in the PRD (source shown in `listmodels`) covers this.
- **`reload()` runs concurrently with request handling.** `_build_maps` must build new maps and swap them in one assignment. It must not mutate maps a request may be reading. This needs a test.
- **Two places describe a model.** The overlay and the bundled snapshot can disagree. The audit has to check the overlay against the facts and flag pins.
- **A new outbound request at startup** to each enabled provider's catalog. It is keyless for OpenRouter and uses the user's own key for Gemini, OpenAI and OpenCode Go. It must respect proxy settings and stay off when the user disables it.

**Neutral**
- `scripts/generate_model_entries.py :: entry_from_facts` and `scripts/audit_model_registry.py :: openrouter_index` move into a runtime module (for example `providers/registries/catalog.py`) so the runtime and the scripts share one mapping. They are stdlib-only today, which suits that.

## Alternatives considered

- **Keep static files and run the audit more often.** This leaves a human merge between the catalog and the user. #233 showed that the automation can stall unnoticed for a week.
- **Fully dynamic, with no overlay.** Auto mode would lose its ranking (`intelligence_score`), and users would lose aliases. Rejected because the judgment fields have no source.
- **Fetch synchronously at startup.** It is simpler, but startup would then depend on the network and fail or hang offline. Rejected: an MCP server that fails to start is worse than one with day-old limits.
- **Use models.dev as a universal fallback source.** It covers more providers, but it was wrong four ways in one week. Rejected for runtime; kept for the audit.

## Decision provenance

Decisions 3 (refresh on every startup), 6 (on by default), 7 (disable withdrawn models) and 8 (audit opens an issue) were Lauri's answers on 2026-11-17 to the open questions in the first draft. The PRD § Decisions records them.
