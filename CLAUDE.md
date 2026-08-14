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
uv run python communication_simulator_test.py --quick   # live MCP end-to-end
```

Integration tests use the free `local-llama` model (`ollama serve && ollama pull llama3.2`, `export CUSTOM_API_URL=http://localhost:11434`), or run `./run_integration_tests.sh`.

Simulator options: `--list-tests`, `--individual <name>`, `--verbose`. After code changes, restart your Claude session for the running MCP server to pick them up.

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
