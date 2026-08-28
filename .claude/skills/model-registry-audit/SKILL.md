---
name: model-registry-audit
description: Sweep conf/*_models.json for deprecated, stale, and missing models against live provider catalogs, then apply the config edits. Use when refreshing PAL's model registry, acting on a model-registry drift PR, adding a newly released model, or removing one a provider withdrew.
allowed-tools: Read, Edit, Write, Grep, Glob, Bash(just *), Bash(uv run *), Bash(python3 *), Bash(curl *), Bash(jq *), Bash(git *), Bash(gh *), WebFetch, WebSearch, TodoWrite
---

# Model Registry Audit

PAL exposes models through eight `conf/*_models.json` files. Provider catalogs
change weekly; those files change when someone remembers. This skill closes the
gap.

The split is deliberate: **`scripts/audit_model_registry.py` finds the drift,
this skill decides what to do about it.** The script never edits a config,
because every interesting decision in this domain — is that model really gone,
does this new one deserve the `pro` alias, what intelligence_score slots it
correctly against its siblings — is judgment a diff cannot encode.

## Run the audit

```
just models-audit
```

Narrow to one provider, or work offline from the cached catalogs:

```
just models-audit-one openrouter_models.json
just models-audit-offline
```

The report ends in a parseable rollup:

```
DEPRECATED=19
STALE=21
ALIAS_COLLISION=0
ORPHAN_REF=3
SCHEMA=0
MISSING=47
ACTIONABLE=43
STATUS=DRIFT
```

`ACTIONABLE` excludes `MISSING` on purpose — candidate additions are ranked by
release date, not by whether they are worth exposing, so a nonzero `MISSING`
is the normal state and must never gate CI.

`ORPHAN_REF` is the one finding that is not about the config at all. Providers
pin canonical ids *outside* the registry — `PRIMARY_MODEL`, `FALLBACK_MODEL`,
and the per-category preference lists in `get_preferred_model` — as plain
strings. Remove a model from the config and those keep pointing at it, so
category routing resolves to nothing while every test still passes: the suite
exercises the resolver, not the constants.

> Found 2026-08-28: `providers/xai.py` named `grok-4-1-fast-reasoning` and
> `grok-4` as PRIMARY/FALLBACK after xAI retired both, and `providers/openai.py`
> listed the shut-down `gpt-5-codex` in three preference lists.

So when you remove a model, grep the provider module in the same edit — or just
re-run the audit, which now does it for you.

## The one trap that matters: absence is not withdrawal

Two catalogs back the audit, and they carry **different authority**:

| Catalog | Covers | A model's absence means |
|---|---|---|
| `openrouter.ai/api/v1/models` | `openrouter_models.json` | **Confirmed gone.** This is the provider's own live serving list. It also carries `expiration_date`, the only explicit deprecation signal either catalog has. |
| `models.dev/api.json` | gemini, openai, xai, opencode zen | **Review only.** Community-maintained. It lags new releases and omits live models outright. |

The script marks every models.dev-derived finding `[review]`. Treat that marker
as a hard gate: **never delete a configured entry on a `[review]` finding
alone.** Confirm against the provider's own documentation first —
`WebFetch` the provider's model page, or hit its `/v1/models` endpoint with a
key if one is available. Deleting a working model because a community catalog
had not indexed it yet is the failure mode this skill exists to prevent, and it
looks exactly like diligent housekeeping in the diff.

### A vendor's own deprecation page outranks both catalogs

Neither catalog carries Google's or OpenAI's shutdown dates, and both lag the
vendors in opposite directions. Before removing anything, read the vendor page:

| Provider | Authoritative source |
|---|---|
| Gemini | Google's deprecations page — reachable via the `gemini-api-docs` MCP (`gemini_search_docs` for "deprecations"). It carries shutdown dates neither catalog has. |
| OpenAI | `https://developers.openai.com/api/docs/deprecations` — a table of shutdown dates and named replacements. |
| xAI | `https://docs.x.ai/docs/models` — a plain list of what is served. |

Two directions of disagreement, both observed on 2026-08-28:

- **The catalog says live, the vendor says dead.** OpenRouter listed
  `openai/gpt-5.1-codex` with no `expiration_date` a month after OpenAI shut it
  down (announced 2026-04-22, shutdown 2026-07-23). An aggregator's listing is
  evidence about the aggregator, not about the origin API — which is why the
  audit compares each config against *its own* provider's catalog rather than
  pooling them.
- **The catalog says nothing, the vendor says deprecated.** `gemini-3.1-flash-lite`
  is present and unremarkable in both catalogs; Google's page gives it a
  shutdown of 2027-05-07 and names `gemini-3.5-flash-lite` as its replacement.

An announced-but-future shutdown is not a removal. Keep the entry, put the date
and replacement in its `description`, and let the alias migrate first.

### Check the provider-id mapping before believing a mass deprecation

Each config maps to one catalog slice in `TARGETS`. A **wrong mapping reads
exactly like a provider withdrawing half its lineup**, because every configured
model is genuinely absent from the slice being searched — the negative is real,
it just answers a different question.

> Observed 2026-08-28: `opencode_go_models.json` was mapped to models.dev's
> `opencode` provider (93 models). The correct slice is `opencode-go` (32
> models) — models.dev carries both, and they are different gateways. The wrong
> mapping reported six live models as withdrawn and called five genuinely-live
> releases fabrications.

The tell is the shape of the result: a scattering of deprecations across a file
is ordinary churn, but a whole model *family* vanishing at once is a mapping
bug until proven otherwise. Confirm the slice contains models you know are
live before acting on any of it:

```
python3 -c "import json;d=json.load(open('.cache/model-catalogs/modelsdev.json'));print([k for k in d if 'opencode' in k])"
```

Three configs cannot be machine-verified at all, and the script says so rather
than reporting them clean:

- **DIAL** — enterprise aggregator, no public keyless catalog. Audit it by
  checking whether the *underlying* models (`o3-2025-04-16`,
  `claude-sonnet-4.1`, `gemini-2.5-pro-preview-03-25`) are still current at
  their origin providers, and treat every conclusion as advisory.
- **Custom** — user-supplied local endpoints (Ollama, vLLM). Whatever the
  operator runs is correct by definition.
- **Azure** — deployment names are chosen per-subscription by the operator.

## Applying findings

### Removing a deprecated model

1. **Confirm the finding's confidence.** `confirmed` → proceed.
   `[review]` → verify against provider docs first, per above.
2. **Rehome the aliases.** An alias is the user-facing contract; deleting
   `x-ai/grok-4` silently takes `grok` with it. Move the alias to the
   replacement in the same edit, or the removal is a regression for anyone
   typing the short name.
3. **Check the blast radius** before deleting. Model ids are referenced
   outside `conf/`:

   ```
   grep -rn "MODEL_ID" providers/ tools/ utils/ docs/ tests/ .env.example README.md
   ```

   Ignore two paths: `CHANGELOG.md` (immutable record, release-please owns it)
   and `tests/*cassettes*/` (recorded HTTP fixtures — a cassette naming a
   retired model is a historical recording, not a live reference).

   Docs need a distinction the grep will not draw for you: `gpt-5.1-codex` (the
   direct OpenAI id, shut down) and `openai/gpt-5.1-codex` (the OpenRouter slug,
   still served) are different facts about different providers. Rewrite the
   first, leave the second.
4. **Run the tests that assert on model names**:

   ```
   uv run pytest tests/test_supported_models_aliases.py tests/test_model_resolution.py tests/test_listmodels.py tests/test_alias_target_restrictions.py tests/test_openrouter_registry.py -q
   ```

### Adding a model

Every field must come from the catalog or the provider's own docs. **Never
infer a context window from the model's name or from a sibling** — a
fabricated `context_window` is silently wrong at runtime and produces
truncation bugs far from the config that caused them.

Pull the real numbers:

```
python3 -c "import json;d=json.load(open('.cache/model-catalogs/modelsdev.json'));print(json.dumps(d['openai']['models']['gpt-5.6'],indent=1))"
```

Then fill the entry against
`providers/shared/model_capabilities.py :: ModelCapabilities`:

| Field | Source |
|---|---|
| `context_window`, `max_output_tokens` | catalog `limit.context` / `limit.output` (models.dev) or `context_length` / `top_provider.max_completion_tokens` (OpenRouter) |
| `supports_images` | catalog `attachment`, or `modalities.input` contains `image` |
| `supports_function_calling` | catalog `tool_call` |
| `supports_extended_thinking` | catalog `reasoning` |
| `supports_temperature` | catalog `temperature`. Reasoning models that reject the parameter get `false` **plus** a `temperature_constraint` of `fixed` |
| `use_openai_response_api` | `true` only for models that require OpenAI's `/responses` endpoint (the Pro reasoning tier) |
| `intelligence_score` | judgment — see below |
| `allow_code_generation` | judgment — `true` only for a model more capable than the CLI driving PAL |

A field outside `CAPABILITY_FIELD_NAMES` is **dropped silently** by
`providers/registries/base.py`. The audit's `SCHEMA` check catches this; a
passing test suite will not.

### Scoring `intelligence_score`

It is a 1–20 human-curated rank driving auto-mode ordering, so it is only
meaningful *relative to the models already in that file*. Read the file's
existing scores first and slot the new entry between the two it belongs
between. Do not restart the scale, and do not import a score from another
provider's file — the ladders are calibrated independently.

### Choosing aliases

Aliases are the reason someone types `pro` instead of
`gemini-3.1-pro-preview`, so they should track the current best model in a
tier rather than a specific version. Two rules:

- **No collision within a file.** The audit's `ALIAS_COLLISION` check enforces
  this; a duplicate resolves unpredictably.
- **Generic aliases move with the frontier.** When `gemini-3.7-flash`
  supersedes `gemini-3.5-flash`, `flash` should follow it, and the superseded
  model keeps only its version-pinned alias (`flash3.5`). Leaving `flash` on
  the old model is how a registry silently rots while every test still passes.

Across files the same alias legitimately appears more than once (`opus` in
both `openrouter_models.json` and `dial_models.json`) — `providers/registry.py`
resolves those by provider priority. That is not a collision.

## Verify before committing

```
just models-audit          # the finding you fixed should be gone
just check                 # lint, format, type check, unit tests
```

Confirm the fix landed by re-reading the audit, not by the edit succeeding.
`STATUS=CLEAN` is the goal for `ACTIONABLE` findings; `MISSING` will still be
nonzero and that is correct.

## The scheduled workflow

`.github/workflows/model-registry-audit.yml` runs this audit weekly and opens
`chore/model-registry-drift` with the report committed to
`docs/model-audit/`. It deliberately **does not edit any config** — it stages
the findings so a human or this skill makes the calls. Rerunning updates the
same branch rather than stacking a new PR each Monday.

To act on such a PR: check out the branch, read
`docs/model-audit/latest-audit.txt`, and work through the sections above.
