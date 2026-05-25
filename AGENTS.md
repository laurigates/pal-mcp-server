# Repository Guidelines

Dependencies live in `pyproject.toml` (deps + `[dependency-groups.dev]`) and are pinned in `uv.lock`. Read `CLAUDE.md` and `CLAUDE.local.md` (if present) before making non-trivial changes.

## Project Structure & Module Organization

PAL MCP Server centers on `server.py`, which exposes MCP entrypoints and coordinates multi-model workflows. Feature-specific tools live in `tools/`, provider integrations in `providers/`, and shared helpers in `utils/`. Prompt and system context assets stay in `systemprompts/`, while configuration templates live under `conf/` and container assets in `docker/`. Unit tests sit in `tests/`; simulator-driven scenarios and log utilities are in `simulator_tests/` with the `communication_simulator_test.py` harness. Authoritative documentation and samples live in `docs/`, and runtime diagnostics rotate in `logs/`.

## Build, Test, and Development Commands

- `uv sync --group dev` — install/refresh the dev environment from `uv.lock`.
- `./run-server.sh` — bootstrap the MCP server (handles cross-platform install, `.env`, and Claude registration).
- `./code_quality_checks.sh` — run ruff (lint + format), ty (type-check), and the unit test suite.
- `uv run pytest tests/ -v -m "not integration"` — unit tests only.
- `uv run python communication_simulator_test.py --quick` — smoke-test orchestration across tools and providers.
- `./run_integration_tests.sh [--with-simulator]` — exercise provider-dependent flows against Ollama (free) or remote models.

Individual test commands:

```bash
uv run pytest tests/test_auto_mode_model_listing.py -q
uv run pytest -q
```

No venv activation needed — `uv run` resolves against the locked environment automatically.

## Coding Style & Naming Conventions

Target Python ≥ 3.10. Lint and format with ruff (120-char line limit; pycodestyle, pyflakes, bugbear, comprehensions, pyupgrade, isort). Type-check with `ty`. Prefer explicit type hints, snake_case modules, and imperative docstrings. Extend workflows by defining hook or abstract methods instead of `hasattr()`/`getattr()` checks — inheritance-backed contracts keep behavior discoverable and testable.

## Testing Guidelines

Mirror production modules inside `tests/` and name tests `test_<behavior>` or `Test<Feature>` classes. Run `uv run pytest tests/ -v -m "not integration"` before every commit; add `--cov=. --cov-report=html` for coverage-sensitive changes. Use `uv run python communication_simulator_test.py --verbose` or `--individual <case>` to validate cross-agent flows; reserve `./run_integration_tests.sh` for provider or transport modifications. Capture relevant excerpts from `logs/mcp_server.log` or `logs/mcp_activity.log` when documenting failures.

## Commit & Pull Request Guidelines

Follow Conventional Commits: `type(scope): summary`, where `type` is one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, or `chore`. Use `feat!:` or `BREAKING CHANGE:` for major bumps. `release-please` parses these to drive automated version bumps and `CHANGELOG.md` updates on push to `main`. Keep commits focused; reference issues or simulator cases where useful. Pull requests should outline intent, list validation commands run, flag configuration or tool toggles, and attach screenshots or log snippets when user-visible behavior changes.

## GitHub CLI Commands

The GitHub CLI (`gh`) streamlines issue and PR management directly from the terminal.

### Viewing issues

```bash
gh issue view <issue-number>                                   # current repo
gh issue view <issue-number> --repo owner/repo-name            # explicit repo
gh issue view <issue-number> --comments                        # with comments
gh issue view <issue-number> --json title,body,author,state,labels,comments
gh issue view <issue-number> --web                             # open in browser
```

### Managing issues

```bash
gh issue list
gh issue list --label bug --state open
gh issue create --title "Issue title" --body "Description"
gh issue close <issue-number>
gh issue reopen <issue-number>
```

### Pull request operations

```bash
gh pr view <pr-number>
gh pr list
gh pr create --title "PR title" --body "Description"
gh pr checkout <pr-number>
gh pr merge <pr-number>
```

Install GitHub CLI: `brew install gh` (macOS) or see <https://cli.github.com>.

## Security & Configuration Tips

Store API keys and provider URLs in `.env` or your MCP client config; never commit secrets or generated log artifacts. Use `./run-server.sh` to regenerate environments and verify connectivity after dependency changes. When adding providers or tools, sanitize prompts and responses, document required environment variables in `docs/`, and update `claude_config_example.json` if new capabilities ship by default.
