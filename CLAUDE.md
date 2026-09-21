# Claude Development Guide for PAL MCP Server

Toolchain is uv + ruff + ty + pytest, all configured in `pyproject.toml`. See `AGENTS.md` for project layout and coding style.

## Setup

```bash
uv sync --group dev
```

Creates `.venv/` from `uv.lock`. No activation needed — prefix commands with `uv run`.

## Quality gate

```bash
./code_quality_checks.sh
```

Runs `uv lock --check`, sync, `ruff check --fix`, `ruff format`, `ruff check`, `ty check`, and unit tests. Must pass before commit.

Individual steps:

```bash
uv run ruff check --fix .                    # lint
uv run ruff format .                         # format
uv run ty check .                            # type check (pre-1.0, advisory)
uv run pytest tests/ -m "not integration"    # unit tests
```

Pre-commit hooks (`ruff`, `ruff-format`, `conventional-pre-commit`):

```bash
uv run pre-commit install
```

## Tests

```bash
uv run pytest tests/ -m "not integration"          # unit (default, no API keys)
uv run pytest tests/ -m "integration"              # needs Ollama + CUSTOM_API_URL
uv run python communication_simulator_test.py --quick   # live MCP end-to-end (needs a Gemini key)
uv run python communication_simulator_test.py --ci      # the subset CI runs, any single provider
```

Integration tests use the free `local-llama` model — the model `conf/custom_models.json` gives that alias (`ollama serve && ollama pull <that model>`, `export CUSTOM_API_URL=http://localhost:11434/v1`), or run `./run_integration_tests.sh`. The `/v1` suffix is load-bearing: the Custom provider passes the URL to the OpenAI client unchanged, and Ollama serves its OpenAI-compatible API only under `/v1`.

Simulator options: `--list-tests`, `--individual <name>`, `--verbose`. After code changes, restart your Claude session for the running MCP server to pick them up.

`--ci` selects the scenarios marked `provider_agnostic` on their test class: they ask for whatever `SIMULATOR_MODEL` names instead of hard-coding `flash`, so they run against any single provider with no key. A new scenario joins the set by declaring `provider_agnostic = True` and taking its model from `self.simulator_model`; `tests/test_simulator_ci_mode.py` rejects a model name written in literally.

`.github/workflows/simulator.yml` runs that set in two tiers, since the model is ~97% of the runtime and 0% of the assertions ([#143](https://github.com/laurigates/pal-mcp-server/issues/143)): a **stub** tier (`simulator_tests/stub_provider.py`, ~20s, every PR) and an **ollama** tier (`llama3.2:1b`, ~25m, `main` + nightly + PRs labelled `simulator-full`). The stub is deliberately strict — wrong path, unknown model, malformed `messages` and streaming all get refused — because a stub that accepts anything tests nothing; relax it and CI stops catching what it refuses. See `docs/testing.md` for both invocations.

## Model registry

`conf/*_models.json` drifts from provider catalogs silently. `just models-audit` diffs it against OpenRouter's live list and models.dev; the `model-registry-audit` skill covers acting on the findings.

## Tool surface

Each capability keeps its own MCP tool name. Clients that defer schema loading (Claude Code among them) show tool names first and fetch a tool's `inputSchema` only when the agent decides to call it, so on those clients the name is the whole routing signal. A blind routing benchmark ([#91](https://github.com/laurigates/pal-mcp-server/issues/91)) put 15 fixed requests through six candidate surfaces with names only: the 19-name surface routed 15/15, every surface that folded names behind an enum on a generic tool lost 4 to 6.5 points, and 38 of those 40 misses were the agent concluding the capability was absent and falling back to its own tools.

- A new capability gets a new tool, not an enum value on an existing one. An enum mode is reachable only by a client that has already loaded that tool's schema.
- Prevent over-triggering with an anti-trigger sentence in the description ("Not for edits you can make yourself"), not by hiding the tool. The benchmark's negative control held on every surface that carried one.
- Consolidation is still right when a capability is genuinely a mode of another tool rather than a distinct instrument a user would ask for by name.
- Descriptions are the eager channel a deferring client pays on every session; schemas are fetched per call. Keep descriptions short and every sentence in them behaviour-bearing.

## Server

```bash
./run-server.sh          # bootstrap + register with Claude
./run-server.sh -f       # follow logs
tail -f logs/mcp_server.log
```

## Conventions

- **Conventional commits** (`feat:`, `fix:`, `chore:`, `feat!:`) — release-please parses these to bump the version and write `CHANGELOG.md`.
- **Never hand-edit** `CHANGELOG.md` or `version` in `pyproject.toml`; release-please owns them. `config.__version__` reads from package metadata.
- **`uv sync` silently re-locks a stale lockfile.** Run `./code_quality_checks.sh` before a bare `uv sync` so the lockfile check sees the drift. To fix drift use `uv lock` (minimal diff), not `uv lock --upgrade`.
- **uv is pinned in three places** that must move together: `[tool.uv] required-version`, `version:` on `setup-uv` in `.github/workflows/`, and the base image in `Dockerfile`. Nothing bumps them automatically.
