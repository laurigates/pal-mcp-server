# Claude Development Guide for PAL MCP Server

This file is the working reference for developing PAL MCP Server with Claude Code. Commands here assume the uv-based toolchain configured in `pyproject.toml`.

## Toolchain at a Glance

| Concern | Tool | Command |
|---|---|---|
| Python version | uv-managed CPython, pinned in `.python-version` | `uv python install` |
| Dependencies | uv + `pyproject.toml` + `uv.lock` | `uv sync --group dev` |
| Lint | ruff | `uv run ruff check .` |
| Format | ruff format | `uv run ruff format .` |
| Type check | ty (Astral) | `uv run ty check .` |
| Unit tests | pytest | `uv run pytest tests/ -m "not integration"` |
| Release | release-please (conventional commits) | automated on push to `main` |

Legacy stack (pip, black, isort, python-semantic-release, `requirements*.txt`, `pytest.ini`) has been retired. `pyproject.toml` is the single source of configuration.

## First-Time Setup

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the pinned Python and sync the dev environment
uv sync --group dev
```

`uv sync` creates `.venv/` and installs the project plus dev dependencies from `uv.lock`. No `source .venv/bin/activate` needed — prefix commands with `uv run` and uv resolves them against the locked environment.

> The MCP server bootstrap (`./run-server.sh`) uses the same uv-managed `.venv/` as day-to-day development. It requires uv on PATH and exits if it is missing, rather than provisioning its own environment. (The stale `.pal_venv` entries in ruff's `extend-exclude` and `.pre-commit-config.yaml` are harmless leftovers.)

## Code Quality

### One-shot quality script

```bash
./code_quality_checks.sh
```

Runs (in order): `uv lock --check`, `uv sync --group dev`, `ruff check --fix`, `ruff format`, `ruff check` (verify), `ty check`, `pytest -m "not integration"`. Must pass 100% before commit.

The lockfile check is deliberately first: `uv sync` re-locks silently when the lockfile is stale, so running it earlier would repair the drift in your working tree and leave the check passing on a fix you never staged.

### Individual checks

```bash
uv run ruff check .                     # lint
uv run ruff check --fix .               # lint + auto-fix
uv run ruff format .                    # format
uv run ruff format --check .            # format check (CI-style)
uv run ty check .                       # type check
uv run pytest tests/ -m "not integration"   # unit tests
```

### Pre-commit hooks

```bash
uv run pre-commit install               # one-time
uv run pre-commit run --all-files       # manual run
```

Configured hooks: `ruff` (with `--fix`), `ruff-format`, `conventional-pre-commit` (commit-msg).

## Testing

### Unit tests (default — fast, no API keys)

```bash
uv run pytest tests/ -v -m "not integration"
uv run pytest tests/test_refactor.py -v
uv run pytest tests/test_refactor.py::TestRefactorTool::test_format_response -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=html -m "not integration"
```

### Integration tests (free, requires local Ollama)

```bash
# Setup: install Ollama, start service, pull a model
ollama serve
ollama pull llama3.2
export CUSTOM_API_URL="http://localhost:11434"

# Run
./run_integration_tests.sh

# Or directly
uv run pytest tests/ -v -m "integration"
```

Integration tests use the `local-llama` model — free to run unlimited times. Excluded from `code_quality_checks.sh` to keep that fast.

### Simulator tests (live MCP end-to-end)

```bash
uv run python communication_simulator_test.py             # all
uv run python communication_simulator_test.py --quick     # 6 essential tests
uv run python communication_simulator_test.py --list-tests
uv run python communication_simulator_test.py --individual basic_conversation
uv run python communication_simulator_test.py --individual memory_validation --verbose
```

**Quick mode (~6 tests covering critical paths)**:
- `cross_tool_continuation`, `conversation_chain_validation`, `consensus_workflow_accurate`, `codereview_validation`, `planner_validation`, `token_allocation_validation`.

**Important**: after any code changes, restart your Claude session for them to take effect in the running MCP server.

## Server Management

### Setup / refresh the MCP server registration

```bash
./run-server.sh         # bootstrap + register with Claude
./run-server.sh -f      # follow logs
```

`run-server.sh` handles the Python install, dependency sync into `.venv/`, `.env` creation, and MCP client registration. It requires uv and syncs with `--frozen`, so it installs exactly what `uv.lock` records rather than re-locking a fresh clone.

### Logs

```bash
tail -f logs/mcp_server.log                      # full activity
tail -f logs/mcp_activity.log                    # tool calls only
grep "ERROR" logs/mcp_server.log | tail -20
tail -f logs/mcp_activity.log | grep -E "(TOOL_CALL|TOOL_COMPLETED|ERROR|WARNING)"
```

Programmatic access for tests:

```python
from simulator_tests.log_utils import LogUtils
recent_logs = LogUtils.get_recent_server_logs(lines=500)
errors = LogUtils.check_server_logs_for_errors()
matches = LogUtils.search_logs_for_pattern("TOOL_CALL.*debug")
```

## Development Workflow

### Before changes
1. `./code_quality_checks.sh` (baseline — syncs the environment itself)
2. `tail -n 50 logs/mcp_server.log` (server health)

Don't run `uv sync` first. It re-locks silently, which defeats the script's lockfile check — the script syncs for you, after that check.

### After changes
1. `./code_quality_checks.sh`
2. `uv run python communication_simulator_test.py --quick`
3. `tail -n 100 logs/mcp_server.log` (check for regressions)
4. Restart Claude session

### Before commit / PR
1. `./code_quality_checks.sh` final
2. `./run_integration_tests.sh` (or `--with-simulator` for the full sweep)
3. Verify everything passes 100%
4. Use a conventional commit (`feat:`, `fix:`, `chore:`, etc.) — `release-please` parses these

## Release Process

Releases are automated by **release-please**:

1. Commit with conventional-commit messages (`feat:` minor, `fix:` patch, `feat!:` / `BREAKING CHANGE:` major).
2. Push to `main`.
3. `release-please.yml` opens (or updates) a release PR with the proposed version bump and `CHANGELOG.md` diff.
4. Review and merge the release PR. release-please tags the release (`v<version>`) and updates `pyproject.toml`'s version.
5. `config.__version__` is derived dynamically from `pyproject.toml` via `importlib.metadata` — no manual sync needed.
6. On release, `release-please.yml`'s `publish-pypi` job builds with `uv build` and publishes to PyPI; the `v<version>` tag push makes `container.yml` promote the container image.

**Don't manually edit**: `CHANGELOG.md`, the `version` field in `pyproject.toml`. release-please owns them.

Authentication uses the **laurigates-release-please GitHub App** (not a PAT). `release-please.yml` is a thin caller for `laurigates/.github/.github/workflows/reusable-release-please.yml@main`, which mints the token via `actions/create-github-app-token`. The caller passes the `RELEASE_PLEASE_APP_ID` variable as the workflow's `app-id` input and the `RELEASE_PLEASE_PRIVATE_KEY` secret as `APP_PRIVATE_KEY` — both pushed by `gitops` to repos flagged `release_please = true`.

## Continuous Integration

Every workflow with an org-wide equivalent is a thin caller for `laurigates/.github` (`@main`):

| Workflow | Reusable workflow called |
|---|---|
| `claude.yml` / `claude-code-review.yml` | `reusable-claude.yml` / `reusable-claude-review.yml` |
| `release-please.yml` | `reusable-release-please.yml` (+ a local `publish-pypi` job) |
| `container.yml` | `reusable-container-build.yml` / `reusable-container-release.yml` |
| `security.yml` | `reusable-security-owasp.yml` / `reusable-security-secrets.yml` |
| `quality.yml` | `reusable-quality-code-smell.yml` |
| `enforce-conventional-commits.yml` | `reusable-enforce-conventional-commits.yml` |
| `renovate.yml` | `reusable-renovate.yml` |
| `clear-autorelease-labels.yml` | `reusable-clear-autorelease-labels.yml` |

Two workflows stay inline because nothing upstream covers them:

- **`test.yml`** — there is no Python/uv reusable workflow, so the pytest matrix and ruff lint jobs live here. Its `lint` job also runs `uv lock --check`.
- **`release-lock.yml`** — release-please bumps `pyproject.toml`'s version without relocking, and `uv.lock` records that version, so every release PR would otherwise fail `uv lock --check`. This workflow runs `uv lock` on the release PR and commits the result using the release App token (a `GITHUB_TOKEN` push would not re-trigger the failed check).

**When merging a release PR, wait for `Container` to go green.** The relock commit supersedes the first container build, and the replacement starts from a cold cache. Merging before it finishes means `:next-<version>` never gets pushed, so the tag-push release job finds nothing to promote and falls back to a full rebuild — a slower release, still green, easy to miss.

Callers declare `permissions:` explicitly because the repo's default workflow permissions are read-only, and the reusable workflows' declared scopes are capped by the caller's.

## Troubleshooting

### Lockfile drift or stale env
```bash
uv lock                                   # fix a red "Verify lockfile is current"
uv sync --group dev --reinstall           # recreate the venv from uv.lock
uv lock --upgrade                         # deliberate dependency refresh — NOT a drift fix
```

`uv lock` does a minimal update and is the same command `release-lock.yml` runs, so it produces the one-line diff CI is asking for. Reach for `--upgrade` only when you actually want every dependency re-resolved to the newest allowed version — using it to clear drift buries a one-line fix in a lockfile-wide diff.

Note that `uv sync` silently re-locks when the lockfile is stale, so it repairs drift in your working tree without saying so. That is why `code_quality_checks.sh` runs `uv lock --check` *before* syncing: otherwise the script would fix the problem locally, report success, and leave you to commit everything except the fix.

### Lint or format issues
```bash
uv run ruff check --fix .                 # auto-fix what's auto-fixable
uv run ruff format .                      # apply formatter
```

### Type errors
```bash
uv run ty check .                         # full report
uv run ty check path/to/file.py           # narrow scope
```

`ty` is pre-1.0; treat its output as advisory while the type-check suite stabilises.

### Test failures
```bash
uv run pytest tests/ -x -q -m "not integration"        # fail fast
uv run pytest tests/test_foo.py::test_bar -v           # narrow
LOG_LEVEL=DEBUG uv run python communication_simulator_test.py --individual <test_name>
```

### Server issues
```bash
./run-server.sh                           # rebootstrap
grep "ERROR" logs/mcp_server.log | tail -20
which python                              # check which interpreter MCP is using
```

## File Structure

| Path | Purpose |
|---|---|
| `pyproject.toml` | Sole config for project, deps, ruff, ty, pytest |
| `uv.lock` | Pinned dependency graph (committed) |
| `.python-version` | Pinned Python (committed) |
| `release-please-config.json` / `.release-please-manifest.json` | Release automation |
| `code_quality_checks.sh` | One-shot quality runner |
| `run-server.sh` | MCP server bootstrap (cross-platform) |
| `communication_simulator_test.py` | Simulator test harness |
| `simulator_tests/` | Simulator test modules |
| `tests/` | Unit tests (`pytest`) |
| `tools/` | MCP tool implementations |
| `providers/` | AI provider implementations |
| `systemprompts/` | System prompt definitions |
| `logs/` | Rotating server log files |

## Environment Requirements

- **uv** ≥ 0.8.17, < 0.12 (managed dependencies + venv) — enforced by `[tool.uv] required-version`, which uv applies to every project command including `uv sync --frozen`. The ceiling exists because `uv lock --check` in CI fails the moment a newer uv rewrites the lockfile's `revision`; the floor is the version verified to read `revision = 3`.

  **Three places pin uv and must be bumped together**: `[tool.uv] required-version`, `version:` on `setup-uv` in the workflows, and the uv base image in `Dockerfile`. **Assume nothing proposes any of them automatically.** The `setup-uv` inputs carry `# renovate:` annotations, but those only produce PRs if the org Renovate config enables a matching `customManager`, and the pinned value is a range (`0.11.x`) rather than an exact version, which such a manager may match without being able to bump. The Dockerfile pin looks like the one Renovate handles natively, but it is pinned to a discontinued variant — Astral publishes no `*-python3.12-bookworm-slim` tag above 0.9.30 — so the `dockerfile` manager will propose nothing until that image line moves.

  The bound spans four minor series only because of that stale container image. CI and contributors alone would be fine at `>=0.11,<0.12`; migrating the builder to the maintained `trixie-slim` line is what would let the bound narrow to a single minor, which is what makes `uv lock --check` fully trustworthy.
- **Python ≥ 3.10** (transitively required by `mcp`; pinned via `.python-version`)
- **`.env`** with provider API keys (created by `./run-server.sh` on first run)
- **Optional**: Ollama for free integration tests

`requirements.txt` and `requirements-dev.txt` no longer exist — `pyproject.toml` + `uv.lock` are the source of truth.
