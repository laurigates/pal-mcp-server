# Prompt caching: design and measurement

PAL should lean on the model providers' own prompt caches. Most providers reuse the
prefix of a request that matches an earlier request, and bill it at a fraction of the
input price. This document records what stops that today, the changes planned to fix
it, and the measurements that decide when the work is done.

All file references are at `288d446` (release 11.7.0) unless stated otherwise.

## Status

| Stage | Issue | State |
|---|---|---|
| 0 — Measurement | #193 | open |
| A — Stable continuation prefix | #194 | open |
| B — Append-only files and history | #195 | open |
| C — Strict prefix (decide after A/B results) | #196 | deferred |
| D — Provider-specific explicit caching (decide after traffic data) | #197 | deferred |

Decisions taken (2026-10-09): implement Stages 0, A and B; decide Stage C from the
Stage A/B measurements; build explicit caching only for the provider families
`just cache-report` shows are actually used; the local experiment arm uses the local
server with the best per-request cache metrics, which is llama.cpp's `llama-server`
(see [M3](#m3-end-to-end-experiment)).

## How PAL's requests look to a cache today

Every provider request is a system message plus one user message
(`providers/openai_compatible.py:316-338`). Exceptions:

- Gemini concatenates `system + "\n\n" + prompt` into one text part (`providers/gemini.py:169-173`).
- The Responses API path sends the system prompt as a leading `user` message (`providers/openai_compatible.py:348-361`).
- The six workflow tools that put files in the expert prompt (codereview, analyze, precommit, refactor, secaudit, testgen) embed the system prompt in the user message and send no system message.

On a continuation, the user message is the whole conversation, rebuilt from scratch on
every call. In byte order:

| Segment | Code | Changes when |
|---|---|---|
| Tool system prompt (+ LOCALE line, capability addenda) | `tools/simple/base.py:430-435` | Tool, model or LOCALE changes. Byte-identical across turns otherwise. |
| `=== CONVERSATION HISTORY (CONTINUATION) ===`, `Thread: <uuid>`, `Tool:` | `utils/conversation_memory.py:880-882` | Per thread |
| **`Turn N/50`** | `utils/conversation_memory.py:883` | **Every call — the first varying byte of every continuation** |
| Files section: newest-first, each header carries the file's mtime | `utils/conversation_memory.py:560-584, 896-990`; `utils/file_utils.py:488, 515, 538` | A file is added, re-referenced, edited or merely touched |
| Prior turns, absolute numbering | `utils/conversation_memory.py:1007-1063` | Append-only until the history budget overflows; after that the window slides every call |
| Trailer `This is turn N+1`, `=== NEW USER INPUT ===`, follow-up `(K exchanges remaining)` | `utils/conversation_memory.py:1079-1092`; `utils/context_builder.py`; `server.py:1010-1050` | Every call (volatile tail) |

Measured with the strict stub (`simulator_tests/stub_provider.py`) recording request
bodies:

- Consecutive chat continuations share exactly the system prompt plus ~106 characters
  of header: 3,994 characters, about 1,000 tokens. That is under OpenAI's 1,024-token
  minimum and Gemini's 2,048/4,096, so chat continuations get essentially no cache
  hits on models without code generation.
- Models with `allow_code_generation` (36 entries across `conf/`) get a much longer
  chat system prompt: 3,888 + 7,052 characters (`GENERATE_CODE_PROMPT`), about 2.7k
  tokens. That clears OpenAI's minimum, so those models may already cache the system
  prompt across threads.
- **#187** dominates everything else. The check at `tools/simple/base.py:344` looks for
  `=== CONVERSATION HISTORY ===`, which never matches the emitted marker, so the tool
  re-records the whole enhanced prompt as a second user turn and nests the history
  inside itself. With tiny inputs the user message grew 1,888 → 8,847 → 20,441 →
  43,561 characters over four turns. With only the marker fixed it is 1,888 → 3,246 →
  3,697 → 4,334.
- Workflow expert prompts iterate Python sets (`tools/workflow/workflow_mixin.py:365-391`,
  `tools/shared/base_models.py:138-149`), so their bytes differ between server
  processes (different `PYTHONHASHSEED`).
- Consensus splices the stance into the system prompt about 1,100 characters in
  (`systemprompts/consensus_prompt.py:23`), so the same model consulted `for` and
  `against` shares only that much.

**PAL measures nothing today.** No code sets a cache control or reads a cached-token
field. `_extract_usage` reads only prompt, completion and total tokens
(`providers/openai_compatible.py:484-491`). The Responses path calls that same
Chat-Completions-shaped extractor (`:427-428`), so the 18 models with
`use_openai_response_api` (12 OpenAI, 6 OpenRouter) record `input_tokens = 0`.

## What the provider caches need

| Mechanism | Providers | PAL must provide |
|---|---|---|
| Automatic prefix match | OpenAI ≤ 5.5, Azure, xAI, OpenRouter's automatic upstreams (OpenAI, DeepSeek, Grok, Moonshot, Z.AI, Gemini…), OpenCode Go upstreams, Gemini 2.5+ implicit caching, local KV-cache reuse | Byte-stable, append-only requests above the model's minimum |
| Implicit caching at message endings | OpenAI / Azure GPT-5.6+ | Turn boundaries that are message boundaries; one message that grows each turn is never reused |
| Nothing unless marked | Anthropic models via OpenRouter and DIAL | `cache_control` breakpoints on stable segments |
| Routing affinity | OpenRouter `x-session-id`; xAI `x-grok-conv-id` (Chat Completions) or `prompt_cache_key` (Responses); OpenAI `prompt_cache_key`; OpenCode Go `x-opencode-session` (already sent) | A per-thread key that is identical from turn 1 |

Minimum cacheable prefix, from the providers' current documentation:

- OpenAI: 1,024 tokens is documented for GPT-5.6+. For GPT-5.5 and earlier the
  minimum varies with request settings, and GPT-5.5 caches at 2,048-token breakpoints.
- Gemini API: 2,048 tokens for 2.5 Flash/Pro; 4,096 for the 3.x Flash models and 3.1 Pro.
  OpenRouter documents different numbers for the same models, so treat
  Gemini-via-OpenRouter minimums as empirical.
- Anthropic: 512 to 4,096 depending on the model.

Cache-write premiums: GPT-5.6+ writes cost 1.25× input in implicit mode; Anthropic
writes cost 1.25× (5-minute TTL) or 2× (1-hour TTL). Reads are 0.05–0.25× input
depending on provider and model.

## Plan

### Stage 0 — Measurement (#193)

Nothing else can be judged without this, so it lands first.

1. **Usage parsers.** Normalise two keys on `ModelResponse.usage`:
   `cached_input_tokens` and `cache_write_input_tokens`.
   - Chat Completions: `usage.prompt_tokens_details.cached_tokens` and `.cache_write_tokens`
     when present; DeepSeek's native `prompt_cache_hit_tokens` as a fallback.
   - Responses: a dedicated extractor reading `input_tokens`, `output_tokens` and
     `input_tokens_details.cached_tokens` / `.cache_write_tokens`. This also fixes the
     zeroed input/output counts.
   - Gemini: `usage_metadata.cached_content_token_count`, recorded as `0` when the
     field is absent but `usage_metadata` is present — the SDK omits it on a miss.
   - Read every new field behind an `isinstance(v, int)` guard; existing tests build
     responses from bare `Mock()` objects.
   - Show cached tokens on the status line (`utils/progress.py`), e.g.
     `12.4k in (9.8k cached) → 3.1k out`.
2. **Usage for every request path.** Workflow expert calls and consensus consults drop
   usage today. Record each call's own usage, with its model and provider, in a
   per-turn key (consensus accumulates `accumulated_responses` across turns, so a
   cumulative key would double-count).
3. **`just cache-report`** reads `<state-dir>/threads/*.jsonl` and prints per
   (provider, model): field coverage, live hit ratio, write share, effective cost,
   a turns-per-thread histogram, and model-switch / cross-tool turns in a separate bucket.
4. **`just cache-prefix`**: the offline harness and CI gates described under
   [M1](#m1-offline-gates-ci). Gates start as `xfail(strict=True)` and are switched on
   by the PR that makes each pass.
5. **Baseline control calls** (manual, recorded here):
   - one guaranteed-miss call per provider (a fresh nonce prompt), to learn whether a
     miss reports the cached field as absent or `0`;
   - one GPT-5.6+ call to check whether PAL currently pays the write premium without
     reads. PAL sends the system prompt as a `user` message that is never the latest
     message, so implicit mode may write on every call and never read. If confirmed,
     this is a cost bug in today's code and Stage D's GPT-5.6+ item moves up.

### Stage A — Stable continuation prefix (#194)

1. **Fix #187 without its two regressions.**
   - Define the history marker once and import it, or better, key the branch off
     `_original_user_prompt`, which `server.py` sets, rather than a substring of a
     prompt the user controls.
   - Build the history from the thread re-loaded *after* `_record_user_turn_or_raise`
     (`server.py:1152`), so a file attached on a continuation reaches the model on
     that turn rather than the next.
   - Keep on the server-built path what `prepare_prompt` does today on continuations:
     directory expansion, `prompt.txt` handling, and the MCP prompt-size check on the
     original prompt.
   - Regression tests: a continuation that attaches a file, one that attaches a
     directory, and one with an oversize prompt; update `tests/test_large_prompt_handling.py`.
2. **Drop the `Turn N/50` header line** (`utils/conversation_memory.py:883`). The
   trailer already states the turn number after the history.
3. **Per-thread routing key from turn 1.** Set the session id to the thread id as soon
   as the thread is created, before the model call. Add `_build_request` overrides
   (modelled on `providers/opencode_go.py`) for OpenRouter (`x-session-id`), xAI
   (header or `prompt_cache_key` depending on the endpoint PAL uses) and OpenAI
   (`prompt_cache_key`). Never add these to the shared builder: strict OpenAI-compatible
   backends may reject unknown fields. Measure turn-1 hits with and without the key,
   since a per-thread key can reduce cross-thread reuse of a shared system prompt.
4. **Deterministic expert prompts.** Sort, or keep first-seen order for, every
   set-valued field rendered into expert prompts: `relevant_context`, `files_checked`,
   `relevant_files`, the `all_relevant_files` set (`tools/workflow/workflow_mixin.py:365-391`,
   `tools/debug.py:307`) and the image list (`:1635`).
5. **Consensus stance out of the system prompt.** Append the stance block after the
   proposal and context files in the user message.

Cassettes `tests/openai_cassettes/chat_gpt5_continuation.json` and
`chat_cross_step2_gpt5_reminder.json` are replayed by an md5 of the request body
(`tests/http_transport_recorder.py:281-326`), so every history-byte change needs them
re-recorded with `OPENAI_API_KEY`. PR #191 rewrites both; land it first.

### Stage B — Append-only files and history (#195)

1. **Files in first-seen order.** Keep the newest-first walk for budget selection only,
   and render the selected files in the order they first appeared.
2. **No mtime in file headers.** Remove `(Last modified: …)` from all three header sites
   (`utils/file_utils.py:488, 515, 538`). A touch or `git checkout` without a content
   change then no longer invalidates everything after that file.
3. **Files inside the turn that attached them.** Replace the separate files section with
   each file rendered in the turn block that first referenced it, re-read from disk on
   every call as today. An unchanged re-reference renders a one-line pointer; a changed
   file shows its new content in place; a file dropped by the budget becomes a one-line
   marker at its position.
4. **History compaction with hysteresis.** Persist the first rendered turn on the thread
   and keep it fixed while the history fits. When it overflows, drop old turns until the
   history is at most ~60% of the budget, and say "Turns 1–K omitted" before the first
   rendered turn. The prefix is then rewritten once per compaction instead of on every
   call after overflow.

### Stage C — Strict prefix (#196, deferred)

Move the constant end-of-prompt guidance (follow-up instructions without the
remaining-exchanges count, the "continuing / do not repeat" trailer, "Please provide a
thoughtful…") into the system prompt, and render the current input as the next turn
block, with turn 1 rendered the same way. Request *k−1* then becomes a byte prefix of
request *k*, including turn 1 → 2. The cost: those instructions lose their end-of-prompt
position, which may affect answer quality.

Decide after Stage A/B: if `just cache-report` shows most threads are 1–2 turns long,
Stage C's turn-1 → 2 reuse and cross-thread system-prompt hits are where the remaining
value is; if threads run long, A/B already captures most of it.

### Stage D — Provider-specific explicit caching (#197, deferred)

- **GPT-5.6+**: pass history as separate messages (one per turn segment) and place an
  explicit cache breakpoint on the last stable segment. Without Stage C the breakpoint
  is required; implicit mode writes only at the latest message, which is the volatile
  tail.
- **Anthropic via OpenRouter and DIAL**: `cache_control` on the last stable segment
  (at most four breakpoints).
- **Write policy**: skip cache writes on one-shot calls (workflow expert analysis,
  consensus consults), where a 1.25×–2× write premium is never repaid.

Build only for the families that carry real traffic in `just cache-report`, unless the
Stage 0 GPT-5.6+ control call confirms a current cost bug.

Not planned: Gemini explicit `CachedContent` (implicit caching is already on for every
listed Gemini model, and explicit caching adds per-hour storage cost and a lifecycle per
thread); `previous_response_id` (OpenAI-only, breaks on model switches).

## Measurement

### M1: offline gates (CI)

`just cache-prefix [--json]` runs the strict stub in-process, drives scripted scenarios
through `server._dispatch_tool_call` (covering `server.py` and `SimpleTool`, where #187
lives), and records each request body exactly as sent.

Scenarios: chat with no files (S1), the same fixture each turn (S2), a new fixture at
turns 1, 3 and 5 (S3), a fixture edited between turns (S4, informational), a
small-context model that overflows the history budget (S5 — needs a second stub
registry entry with a small `context_window`), consensus `for` then `against` (S6), and
codereview/debug/thinkdeep expert requests with at least two relevant files (S7).

`ser(r)` concatenates, per message, the role, a newline and the text; image parts become
`<img:sha256[:16]>`. For consecutive requests *r_j*, *r_{j+1}* of one scenario,
`L_j = LCP(ser(r_j), ser(r_{j+1}))`.

Gates are structural, so they cannot be passed by sending less:

| Gate | Condition | Switched on by |
|---|---|---|
| **G0 completeness** | In every *r_{j+1}*, each earlier user prompt and each stub reply appears exactly once, and each attached file's content appears exactly once — including files attached on the current turn | Stage A (fails today because of #187) |
| G1 | S1, S2: `L_j` reaches past the end of the last turn block of *r_j* (located by marker) | Stage A |
| G2 | G1's condition on S3 | Stage B |
| G3 | S1–S3: `ser(r_{j+1}).startswith(ser(r_j))` for every *j* | Stage C |
| G4 | S7: request sha256 identical under `PYTHONHASHSEED` 1, 2, 3, with a same-seed control run first and one untouched fixture directory | Stage A |
| G5 | S6: the two consults first differ at the stance block | Stage A |
| G7 | S5: at most 2 pairs over 12 turns where `L_j` falls before the last turn block of *r_j* | Stage B |

Reported but not gated: the simulated cache-hit ratio `SCHR = Σ cacheable(L_j) / Σ tokens(r_{j+1})`,
with `tokens(c) ≈ c // 4` and `cacheable` model-aware (minimum length, and 2,048-token
breakpoints for GPT-5.5). It is an upper-bound proxy, not a prediction of exact
`cached_tokens`.

The stub also returns `usage.prompt_tokens_details.cached_tokens` = the LCP with the
previous request of the same thread, `// 4`. A `provider_agnostic` simulator scenario
asserts that this value reaches the transcript unchanged and grows from turn 3. This adds
a response field only; the stub's refusals are not relaxed.

### M2: live usage

From Stage 0 on, every transcript turn carries the provider's cached and written token
counts. `just cache-report` computes per (provider, model), over requests 3+ of a thread
that used the same tool and model as the previous request:

- **Hit ratio** `Σ cached_input_tokens / Σ input_tokens`. The denominator is the
  provider-normalised total input; the Stage 0 control calls establish, per path,
  whether `prompt_tokens` includes cached tokens (unverified for `anthropic/*` via
  OpenRouter and DIAL).
- **Write share** `Σ cache_write_input_tokens / Σ input_tokens`.
- **Effective cost** `Σ[(input − cached − written)·1 + cached·r_read + written·r_write]`
  relative to the same tokens uncached, using each model's read and write multipliers.

### M3: end-to-end experiment

`just cache-experiment --models … --runs 3 --label <arm>` runs a scripted chat thread
(T-A: 6 turns; pinned fixtures from `288d446` — `providers/gemini.py` and
`utils/token_utils.py` at turn 1, `providers/openai_compatible.py` at turn 4) and a
consensus pair (T-B), in-process with a fresh `PAL_STATE_DIR` per run. A wrapper on
`_call_api` captures each raw provider request and response, so live metrics can be
computed independently of PAL's parser and cross-checked against the transcript.

- **Arms**: baseline = `288d446` + the Stage 0 parser change (plain `288d446` cannot
  report cached tokens and records zero input on Responses models); candidate = the
  branch under test. Arm order is randomised.
- **Contamination**: each run's first user turn carries a per-run nonce after the
  system prompt, so runs cannot hit entries left by earlier runs (OpenAI retention can
  be 24 hours). Run 1 is reported separately from runs 2–3 as a check.
- **Confounds recorded per request**: start time and gap since the previous request,
  the resolved model id, and OpenRouter's upstream `provider`. Pairs further apart than
  the provider's cache lifetime are excluded from realisation.
- **Models**: gpt-5.5; one GPT-5.6+ model; gemini-2.5-flash and one Gemini 3.x Flash;
  one automatic model on OpenRouter; one `anthropic/*` on OpenRouter; one local model,
  served by llama.cpp's `llama-server` (below).

**Local arm.** `llama-server` (Metal; `brew install llama.cpp`) reports the most per
request of the local servers compared on Apple Silicon (2026-10-09):

- `usage.prompt_tokens_details.cached_tokens` in the OpenAI-compatible response;
- `timings.cache_n` (prompt tokens reused), `prompt_n` and `prompt_ms` (prefill of the new part);
- server-wide counters on `/metrics` (with `--metrics`), and slot state on `/slots`.

Run it with one slot (`-np 1`) so every request lands on the slot holding the prefix,
with a context large enough for six turns plus fixtures (`-c 16384` or more), and with
a non-thinking instruct model. Chat templates that strip earlier `<think>` blocks
rewrite the history and break reuse. Point PAL at it through `CUSTOM_API_URL=http://localhost:8080/v1`.

The alternatives, ranked by the same comparison:

- **Ollama ≥ 0.33.3** reports `cached_tokens` on `/v1`; 0.33.2 does not. It runs
  `llama-server` underneath, but exposes no timings or server metrics. Its default
  4,096-token context truncates the front of a six-turn prompt unless
  `OLLAMA_CONTEXT_LENGTH` is raised.
- **MLX-LM** (`mlx_lm.server`) reports `cached_tokens`, but has no timings or metrics.
- **vLLM** on Apple Silicon (the `vllm-metal` plugin) has the most production-like paged
  cache. Its counts are block-aligned, need `--enable-prompt-tokens-details`, and have
  come back empty in reported bugs; this is unverified on Metal. Worth a second arm only
  if a GPU host is used.
- **SGLang** on Apple Silicon needs a source build, and RadixAttention on its MLX
  backend is unverified. Not used.

Reported per model and arm: median and range of the hit ratio for requests 3–6,
predicted SCHR on the recorded bytes, realisation (hit ratio ÷ SCHR), Σ input / cached /
written tokens, effective cost, and wall time.

## Done criteria

Each stage is done when every item below for it holds. Each is a command or a number
recorded in the results table.

**Stage 0**

- `uv run pytest tests/test_openai_compatible_token_usage.py tests/test_gemini_token_usage.py tests/test_openai_provider.py`
  passes with cases that read cached tokens from Chat Completions, Responses (with
  `input_tokens` equal to the fixture value, not 0) and Gemini usage, including a
  Gemini miss with the field absent.
- `just cache-prefix --json` runs in CI; all gates present, failing ones marked
  `xfail(strict=True)`.
- The stub-tier simulator (`communication_simulator_test.py --ci`) passes the cache
  scenario.
- The control-call results (miss representation per provider; GPT-5.6+ write share) are
  recorded below.

**Stage A**

- `rg -n xfail tests/test_prompt_prefix_stability.py` shows no xfail on G0, G1, G4, G5.
- `uv run pytest tests/test_chat_openai_integration.py` passes on re-recorded cassettes.
- In M3 against baseline, for gpt-5.5, gemini-2.5-flash, one OpenRouter automatic model
  and the local model: median hit ratio over requests 3–6 ≥ 0.60, realisation ≥ 0.80,
  and effective cost ≤ baseline. The input-token reduction from the #187 fix is reported
  separately and does not count toward the caching criteria.

**Stage B**

- No xfail on G2 or G7.
- In M3, the T-A dip at turn 4 (third file attached) is gone: the turn-4 hit ratio is
  within 0.1 of turns 3 and 5 for the same models.

## Results

| Date | Commit | Arm | Model | Hit ratio 3–6 (median, range) | SCHR | Realisation | Σ input / cached / written | Effective cost vs uncached |
|---|---|---|---|---|---|---|---|---|

Control calls:

| Date | Provider / model | Miss reports cached field as | Write share (GPT-5.6+) |
|---|---|---|---|

## Unverified provider facts

- Whether OpenCode Go passes upstream cache fields through.
- Whether OpenRouter's `prompt_tokens` includes cached tokens for `anthropic/*`, and the
  same for DIAL.
- Azure `prompt_cache_key` support at the api-version PAL uses.
- Whether xAI on PAL's endpoint returns the cached field.
- The minimum prefix for Gemini models reached through OpenRouter.

## Noticed while investigating (not part of this work)

- Chat's web-search guidance never reaches the model: `prepare_chat_style_prompt`
  keeps only the text after `=== USER REQUEST ===`, and the guidance sits before it
  (`tools/simple/base.py:1076-1080`).
- codereview's expert prompt contains literal `\n` sequences from f-strings.
- `customize_expert_analysis_prompt` in thinkdeep is never called (`tools/thinkdeep.py:285`).
- For workflow tools, the server builds the full conversation history and then discards
  it (`server.py:1195-1205`).
