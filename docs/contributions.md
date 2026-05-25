# Contributing to PAL MCP Server

Thank you for your interest in contributing to PAL MCP Server. This guide covers the development process, coding standards, and how to submit high-quality contributions.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally.
3. **Install uv** if you don't have it: <https://docs.astral.sh/uv/getting-started/installation/>.
4. **Bootstrap the dev environment**:
   ```bash
   uv sync --group dev
   ```
5. **Bootstrap the MCP server** (handles `.env`, Claude registration, cross-platform install):
   ```bash
   ./run-server.sh
   ```
6. **Create a feature branch** from `main`:
   ```bash
   git switch -c feat/your-feature-name
   ```

## Development Process

### 1. Code Quality Standards

All contributions must pass automated checks: ruff lint, ruff format, ty type-check, and the unit test suite.

#### Option 1 — pre-commit (recommended)

```bash
# One-time install
uv run pre-commit install

# Hooks now run automatically on every commit:
# - ruff (lint + --fix)
# - ruff-format
# - conventional-pre-commit (commit-msg)
```

#### Option 2 — comprehensive runner

```bash
./code_quality_checks.sh
```

Runs: `uv sync --group dev` → `ruff check --fix` → `ruff format` → `ruff check` → `ty check` → `pytest -m "not integration"`.

#### Individual commands

```bash
uv run ruff check .                # lint
uv run ruff check --fix .          # lint + auto-fix
uv run ruff format .               # format
uv run ruff format --check .       # format check (CI-style)
uv run ty check .                  # type check
uv run pytest tests/ -xvs -m "not integration"
```

**Important**:
- Every test must pass — zero tolerance for failing tests in CI.
- Both `ruff check` and `ruff format --check` must pass cleanly.
- PRs that fail GitHub Actions are blocked from merging.

### 2. Testing Requirements

#### When to add tests

- **New features must include tests**: add unit tests in `tests/` covering success and error cases.
- **Tool changes require simulator tests**: add `simulator_tests/` covering realistic prompts; validate via server logs.
- **Bug fixes require regression tests**: include a test that would have caught the bug.

#### Test naming conventions

- Unit tests: `test_<feature>_<scenario>.py`
- Simulator tests: `test_<tool>_<behavior>.py`

### 3. Pull Request Process

#### PR title format

PR titles MUST follow Conventional Commits — `release-please` parses them to drive automated version bumps and `CHANGELOG.md` updates.

**Version-bumping prefixes**:
- `feat: <description>` — new feature (MINOR bump)
- `fix: <description>` — bug fix (PATCH bump)
- `perf: <description>` — performance improvement (PATCH bump)
- `feat!: <description>` or include `BREAKING CHANGE:` in the body — breaking change (MAJOR bump)

**Non-bumping prefixes** (visible in changelog as hidden sections):
- `docs:`, `chore:`, `style:`, `refactor:`, `test:`, `build:`, `ci:`

#### PR checklist

Use the [PR template](../.github/pull_request_template.md) and ensure:

- [ ] PR title follows Conventional Commits.
- [ ] Ran `./code_quality_checks.sh` (all checks passed 100%).
- [ ] Self-review completed.
- [ ] Tests added for all changes.
- [ ] Documentation updated as needed.
- [ ] Relevant simulator tests passing (if tool changes).
- [ ] Ready for review.

### 4. Code Style Guidelines

#### Python style

- Format and lint with ruff (120-char line limit, pycodestyle, pyflakes, bugbear, comprehensions, pyupgrade, isort).
- Use type hints on function parameters and returns.
- Add docstrings to public functions and classes.
- Keep functions focused and under 50 lines when reasonable.
- Use descriptive variable names.

#### Example

```python
def process_model_response(
    response: ModelResponse,
    max_tokens: int | None = None,
) -> ProcessedResult:
    """Process and validate a model response.

    Args:
        response: Raw response from the model provider.
        max_tokens: Optional token limit for truncation.

    Returns:
        ProcessedResult with validated and formatted content.

    Raises:
        ValueError: If the response is invalid or exceeds limits.
    """
```

#### Import organisation

Ruff's `I` rule (isort) groups imports into:
1. Standard library
2. Third-party
3. Local application

Run `uv run ruff check --fix .` to sort them automatically.

### 5. Specific Contribution Types

#### Adding a new provider
See [Adding a New Provider](./adding_providers.md).

#### Adding a new tool
See [Adding a New Tool](./adding_tools.md).

#### Modifying existing tools

1. Preserve backward compatibility unless explicitly breaking.
2. Update all affected tests.
3. Update documentation if behavior changes.
4. Add simulator tests for new functionality.

### 6. Documentation Standards

- Update `README.md` for user-facing changes.
- Add docstrings to all new code.
- Update relevant files under `docs/`.
- Include examples for new features.
- Keep documentation concise and clear.

### 7. Commit Message Guidelines

Conventional Commits format:

- First line: `type(scope): brief summary` (50 chars or less).
- Blank line.
- Detailed explanation if needed.
- Reference issues: `Fixes #123`.

Example:

```
feat(gemini): add retry logic to Gemini provider

Implements exponential backoff for transient errors in Gemini API
calls. Retries up to 2 times with configurable delays.

Fixes #45
```

## Common Issues and Solutions

### Lint or format failures

```bash
uv run ruff check --fix .
uv run ruff format .
```

### Test failures

- Check test output for specific errors.
- Run individual tests: `uv run pytest tests/test_specific.py -xvs`.
- Ensure the server environment is set up for simulator tests.

### Import or dependency errors

```bash
uv sync --group dev --reinstall          # recreate env from uv.lock
uv lock --upgrade                         # bump to newest allowed versions
```

## Getting Help

- **Questions**: open a GitHub issue with the "question" label.
- **Bug reports**: use the bug report template.
- **Feature requests**: use the feature request template.
- **Discussions**: use GitHub Discussions for general topics.

## Code of Conduct

- Be respectful and inclusive.
- Welcome newcomers and help them get started.
- Focus on constructive feedback.
- Assume good intentions.

## Recognition

Contributors are recognized in:
- The GitHub contributors page.
- Release notes for significant contributions.
- Special mentions for exceptional work.

Thank you for contributing to PAL MCP Server.
