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

> The MCP server bootstrap (`./run-server.sh`) still maintains its own `.pal_venv/` because it has to provision an environment on machines that may not have uv. Day-to-day development uses `.venv/` via uv.

## Code Quality

### One-shot quality script

```bash
./code_quality_checks.sh
```

Runs (in order): `uv sync --group dev`, `ruff check --fix`, `ruff format`, `ruff check` (verify), `ty check`, `pytest -m "not integration"`. Must pass 100% before commit.

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

`run-server.sh` handles cross-platform Python install, `.pal_venv` provisioning, `.env` creation, and MCP client registration. It already prefers `uv venv` when uv is on PATH.

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
1. `uv sync --group dev` (ensure lockfile-current environment)
2. `./code_quality_checks.sh` (baseline)
3. `tail -n 50 logs/mcp_server.log` (server health)

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
4. Review and merge the release PR. release-please tags the release and updates `pyproject.toml`'s version.
5. `config.__version__` is derived dynamically from `pyproject.toml` via `importlib.metadata` — no manual sync needed.

**Don't manually edit**: `CHANGELOG.md`, the `version` field in `pyproject.toml`. release-please owns them.

Required repo secret: `MY_RELEASE_PLEASE_TOKEN` (PAT with `contents:write` + `pull-requests:write`).

## Troubleshooting

### Lockfile drift or stale env
```bash
uv sync --group dev --reinstall          # recreate the venv from uv.lock
uv lock --upgrade                         # update lockfile to newest allowed versions
```

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

- **uv** ≥ 0.5 (managed dependencies + venv)
- **Python ≥ 3.10** (transitively required by `mcp`; pinned via `.python-version`)
- **`.env`** with provider API keys (created by `./run-server.sh` on first run)
- **Optional**: Ollama for free integration tests

`requirements.txt` and `requirements-dev.txt` no longer exist — `pyproject.toml` + `uv.lock` are the source of truth.
