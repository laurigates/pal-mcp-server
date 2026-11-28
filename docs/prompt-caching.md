# Prompt caching: design and measurement

PAL should lean on the model providers' own prompt caches. Most providers reuse the
prefix of a request that matches an earlier request, and bill it at a fraction of the
input price. This document records what stops that today, the changes planned to fix
it, and the measurements that decide when the work is done.

All file references are at `288d446` (release 11.7.0) unless stated otherwise.

## Status

| Stage | Issue | State |
|---|---|---|
| 0 — Measurement | #193 | code landed; Gemini control calls recorded, the others pending |
| A — Stable continuation prefix | #194 | code and offline gates done; cassette re-record and M3 pending |
| B — Append-only files and history | #195 | code and offline gates done; cassette re-record and M3 pending |
| C — Strict prefix (decide after A/B results) | #196 | deferred |
| D — Provider-specific explicit caching (decide after traffic data) | #197 | GPT-5.6+ message history prototyped behind `PAL_MESSAGE_HISTORY` (2026-11-28): M3 hit ratio 0.94 on gpt-6-luna, against 0.00 for Stage B |

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
| Automatic prefix match | OpenAI ≤ 5.5, OpenRouter's automatic upstreams (OpenAI, DeepSeek, Grok, Moonshot, Z.AI, Gemini…), OpenCode Go upstreams, Gemini 2.5+ implicit caching, local KV-cache reuse | Byte-stable, append-only requests above the model's minimum |
| Implicit caching at message endings | OpenAI GPT-5.6+ | Turn boundaries that are message boundaries; one message that grows each turn is never reused |
| Nothing unless marked | Anthropic models via OpenRouter | `cache_control` breakpoints on stable segments |
| Routing affinity | OpenRouter `x-session-id`; OpenAI `prompt_cache_key`; OpenCode Go `x-opencode-session` (already sent) | A per-thread key that is identical from turn 1 |

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
   (modelled on `providers/opencode_go.py`) for OpenRouter (`x-session-id`) and OpenAI
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

As built (2026-11-23):

- A file's content is read from disk on every call and appears at its first reference,
  so an edited file changes in place there. Later references are one-line pointers.
  A file the budget leaves out is a one-line marker at its first reference.
- The files of the request being answered come after the history's trailer, next to
  NEW USER INPUT. On the next request their turn block takes the position after the
  last turn, and nothing before it changes.
- Compaction persists nothing. `_compaction_start` replays it over the thread's turns
  in order: whenever the kept turns outgrow the budget, the oldest are dropped down to
  60% of it. The result depends only on the turns and the budget, so consecutive
  requests agree on it until the next compaction. The files of dropped turns stay, in
  a block after the "Turns 1–K omitted" note.
- Known limit: 60% hysteresis frees 40% of the budget, so compactions are spaced out
  only while one request (prompt plus reply) stays below about 20% of the history
  budget. Simulated with `_compaction_start` on equal-sized turns: at 10% per request
  the start moved 5 times in 30 requests, never on two consecutive requests; at 30% it
  moved on 18 of 29 requests. This matters for small-context models; #262 tracks it.
- Known limit: the compaction budget is the history budget minus the tokens of every
  embedded file, including the new input's attachments. Once a thread has compacted,
  attaching, editing or evicting a file changes that budget, and `_compaction_start`
  replays against it, so the start can move retroactively and rewrite the prefix from
  the "Turns 1–K omitted" note on. `_compaction_start([5]*30, 90)` is 18 and
  `_compaction_start([5]*30, 85)` is 16. Below the first overflow the turns are
  unaffected. S3 never overflows and S5 attaches no files, so G2 and G7 do not cover it.

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
- **Anthropic via OpenRouter**: `cache_control` on the last stable segment
  (at most four breakpoints).
- **Write policy**: skip cache writes on one-shot calls (workflow expert analysis,
  consensus consults), where a 1.25×–2× write premium is never repaid.

Build only for the families that carry real traffic in `just cache-report`, unless the
Stage 0 GPT-5.6+ control call confirms a current cost bug.

**Prototype (2026-11-28), pulled forward by the probes in [Results](#results).** GPT-5.6+
reuses only the whole previous request, cut at a message boundary, so no layout of one
growing user message can hit, whatever Stages A–C do to its bytes. With
`PAL_MESSAGE_HISTORY=true`, for providers that set `SUPPORTS_MESSAGE_HISTORY` (the OpenAI
provider), `utils/message_history.py` sends a simple-tool continuation as the system
prompt, then each earlier turn's user message exactly as it was sent (stored as
`sent_prompt` on the reply's `model_metadata`) and the reply, then the new input. Request
j+1 is request j plus two messages (`tests/test_message_history_e2e.py`). Earlier turns'
files stay in the message that attached them, so they are read from disk once, when sent.
Falls back to the rendered history for a reply stored without `sent_prompt`, a cross-tool
thread or a chained thread. Not yet handled: compaction when the messages outgrow the
history budget, workflow tools, and an explicit breakpoint.

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
| G7 | S5: every pair where `L_j` falls before the last turn block of *r_j* is a compaction (the "Turns 1–K omitted" note changed), and no two consecutive pairs break | Stage B |

G7 first allowed at most 2 such pairs over 12 turns. That was refuted when Stage B
was built: each S5 request adds about 20% of the turn budget, so 60% hysteresis
compacts on every second request (4 breaks, against 10 at baseline), and only a keep
ratio below about 0.14 passes it, which leaves roughly one turn after a compaction. The
gate now tests the mechanism instead of a count. A sliding window, or compaction
without hysteresis, still fails it, and meta-tests in
`tests/test_prompt_prefix_stability.py` keep both failing.

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
  OpenRouter).
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
| 2026-11-28 | `8b38818`+`d53f53d` | baseline | gpt-6-luna | 0.00 (0.00–0.00) | 0.00 | – | 1,144,825 / 0 / 1,144,801 | 1.25 |
| 2026-11-28 | `84b0a0e` | Stage B | gpt-6-luna | 0.00 (0.00–0.00) | 0.63 | 0.00 | 88,683 / 0 / 88,659 | 1.25 |
| 2026-11-28 | `473fa34` | Stage D prototype | gpt-6-luna | 0.94 (0.55–0.96) | 0.84 | 1.11 | 126,883 / 106,697 / 20,162 | 0.28 |

T-A, runs 2–3, `just cache-experiment --seed 282` (harness `bab5862`, #280), run 1 identical
in ratio. The baseline is `288d446` plus the Stage 0 parser and #278 (branch
`m3/baseline-288d446`). Stage D ran with `PAL_MESSAGE_HISTORY=true`; each request read
back exactly the previous request's write (cached(n) = input(n−1) − 3). Turn 4's 0.55 is
the third fixture entering as new content (~7.8k tokens written), with the prefix before
it fully reused. T-B (consensus) stays at 0.00 in every arm: the prototype covers simple
tools only. Effective cost uses the harness defaults (reads 0.1×, writes 1.25×).
Earlier runs the same day on gpt-6-luna and gemini-3.8-flash (baseline, Stage A, Stage B)
read 0 on all 144 calls; the first of them sent no fixtures, because PAL refuses files
under macOS's `/var` temp directories (fixed in the harness).

Control calls. Each sends a ~8.2k-token prompt that starts with a fresh nonce
(guaranteed miss), then the identical request a few seconds later. Gemini was
called through the `google-genai` SDK, and the response was passed to PAL's
`GeminiModelProvider._extract_usage`. The other providers still need live keys.

| Date | Provider / model | Miss reports cached field as | Repeat: cached / prompt tokens | Write share (GPT-5.6+) |
|---|---|---|---|---|
| 2026-11-23 | Gemini API / gemini-2.5-flash | absent (`None`); PAL records 0 | 8,170 / 8,182 | — |
| 2026-11-23 | Gemini API / gemini-3.8-flash | absent (`None`); PAL records 0 | 6,136 / 8,183 | — |

On both models `prompt_token_count` was the same on the miss and the hit, so it
includes cached tokens, and the hit ratio's denominator needs no correction on
the Gemini API path. Thinking tokens (`thoughts_token_count`) are missing from
PAL's output and total counts: #258.

Offline baseline, `just cache-prefix` on the Stage 0 code: every gate (G0–G5, G7)
fails, as described above. Consecutive chat continuations share 5,799 characters
(`L_j` for S1–S3); the two S6 consults first differ at character 1,116; S5
overflows the history budget and breaks before the last turn block on 10 of 11
pairs. The harness does not build S4 (informational).

Offline, `just cache-prefix` on the Stage A code (2026-11-23): G0, G1, G4 and G5
pass; G2, G3 and G7 still fail, as expected before Stages B and C.

| Scenario | SCHR, Stage 0 | SCHR, Stage A | `L_j` after Stage A |
|---|---|---|---|
| S1 chat, no files | 0.674 | 0.709 | 5,694 · 6,081 · 6,258 |
| S2 same file each turn | 0.479 | 0.677 | 5,694 · 9,058 · 9,411 |
| S3 new file at turns 1, 3, 5 | 0.395 | 0.512 | 5,694 · 6,118 · 11,863 · 6,118 |
| S5 small context, 12 turns | 0.361 | 0.409 | 8 of 11 pairs at ~5,900 |

From turn 2 on, each S1/S2 pair now shares everything through the previous
request's last turn block. The first pair stops at 5,694 characters, where
turn 1 puts the request and turn 2 starts its history block (Stage C). In S3 the pairs that attach a new
file stop at 6,118, where the newest-first file section changes (Stage B). The
S6 consults share 9,640 characters, past the end of the context file at 9,505.

Offline, `just cache-prefix` on the Stage B code (2026-11-23): G0, G1, G2, G4, G5 and
G7 pass; G3 fails, as expected before Stage C.

| Scenario | SCHR, Stage A | SCHR, Stage B | `L_j` after Stage B |
|---|---|---|---|
| S1 chat, no files | 0.709 | 0.709 | 5,694 · 6,081 · 6,258 |
| S2 same file each turn | 0.677 | 0.666 | 5,694 · 8,882 · 9,397 |
| S3 new file at turns 1, 3, 5 | 0.512 | 0.622 | 5,694 · 8,882 · 9,228 · 12,159 |
| S5 small context, 12 turns | 0.409 | 0.533 | breaks at pairs 4, 6, 8, 10 (one per compaction) |

S3's pairs that attach a file now reach past the previous request's last turn
block. S2 drops slightly because each re-reference adds a one-line pointer.

Cache-shape probes (2026-11-28), through PAL's own providers on stage B code, ~14k-token
OpenAI and ~18.7k-token Gemini prompts, one sample per case. M3's first live runs had
read 0 cached tokens on all 144 calls, on every arm, and these probes locate why:

| Case | gpt-6-luna read: cached / input | gemini-3.8-flash read: cached / input |
|---|---|---|
| Exact repeat, 0.5–10 s after the write | 14,438 / 14,441 (every gap) | 16,679–16,683 / ~18,733 (every gap) |
| Last message's tail changed | 0 / 14,441 | 0 |
| Last message extended | 0 / 14,445 | 0 |
| New message appended after the unchanged request | 14,437 / 14,451 | 0 (as a part or a second content) |
| Diverging after the 1st or 2nd of 2–3 messages | 0 | 0 |

- gpt-6-luna writes one cache entry, at the end of the request
  (`input_tokens_details.cache_write_tokens` ≈ input, 30-minute TTL), and reads it only
  when the next request starts with the whole previous request, message for message.
  PAL's continuations change the last message every turn, so every call paid the 1.25×
  write premium and none read: effective input cost 1.25 of uncached on all arms. This
  is the Stage 0 cost bug the GPT-5.6+ control call was meant to detect.
- gemini-3.8-flash hit only on exact repeats; one write in the boundary case reported
  16,683 cached on a fresh nonce. Too few samples to say what it reuses.

## Unverified provider facts

- Whether OpenCode Go passes upstream cache fields through.
- Whether OpenRouter's `prompt_tokens` includes cached tokens for `anthropic/*`.
- The minimum prefix for Gemini models reached through OpenRouter.

## Noticed while investigating (not part of this work)

- Chat's web-search guidance never reached the model: `prepare_chat_style_prompt`
  kept only the text after `=== USER REQUEST ===`, and the guidance sat before it.
  Fixed in #199: the guidance now rides in the user section, so the system prompt
  stays byte-stable.
- codereview's expert prompt contains literal `\n` sequences from f-strings.
- For workflow tools, the server builds the full conversation history and then discards
  it (`server.py:1195-1205`).
