# Testing Guide

This project covers behavior through unit tests (fast, no API keys), simulator tests (real MCP protocol), and integration tests (free local models).

## Running Tests

### Prerequisites

- uv installed: <https://docs.astral.sh/uv/getting-started/installation/>
- Dev environment synced: `uv sync --group dev`
- For simulator/integration tests: MCP server bootstrapped via `./run-server.sh` (use `-f` to follow logs after starting).

### Unit Tests

Fast, isolated tests against function and class behavior. No API keys required.

```bash
uv run pytest -xvs                              # all unit tests, verbose, stop on first failure
uv run pytest tests/test_providers.py -xvs      # specific file
uv run pytest tests/ --cov=. --cov-report=html  # with coverage
```

### Simulator Tests

Simulator tests replicate real-world Claude CLI interactions with the standalone MCP server. They validate the complete end-to-end flow including:

- Actual MCP protocol communication
- Standalone server interactions
- Multi-turn conversations across tools
- Log output validation

**Important**: simulator tests require `LOG_LEVEL=DEBUG` in your `.env` file to validate detailed execution logs.

#### Monitoring logs during tests

The MCP stdio protocol captures stderr during tool execution to prevent interference with the JSON-RPC channel — so tool execution logs land in `logs/` files, not the console.

```bash
# Start server and follow logs in one step
./run-server.sh -f

# Or manually monitor
tail -f -n 500 logs/mcp_server.log         # main server activity
tail -f logs/mcp_activity.log              # tool calls and completions
ls -lh logs/mcp_*.log*                     # check rotation status
```

Log rotation: 20 MB cap per file. The server keeps:
- 10 rotated files for `mcp_server.log` (~200 MB total)
- 5 rotated files for `mcp_activity.log` (~100 MB total)

#### Running all simulator tests

```bash
uv run python communication_simulator_test.py
uv run python communication_simulator_test.py --verbose
uv run python communication_simulator_test.py --keep-logs
```

#### Running individual tests

```bash
uv run python communication_simulator_test.py --individual basic_conversation
uv run python communication_simulator_test.py --individual content_validation
uv run python communication_simulator_test.py --individual cross_tool_continuation
uv run python communication_simulator_test.py --individual memory_validation
```

#### Other options

```bash
uv run python communication_simulator_test.py --list-tests
uv run python communication_simulator_test.py --tests basic_conversation content_validation
```

### Integration Tests (free, local models)

Integration tests use a local Ollama instance, so they're free to run unlimited times.

```bash
# Setup
ollama serve
ollama pull llama3.2
export CUSTOM_API_URL="http://localhost:11434"

# Run
./run_integration_tests.sh
# or directly
uv run pytest tests/ -v -m "integration"
```

Integration tests are excluded from `code_quality_checks.sh` to keep that pipeline fast.

### Code Quality Checks

Before committing, run the quality gate:

```bash
./code_quality_checks.sh
```

Individual commands:

```bash
uv run ruff check .                # lint
uv run ruff check --fix .          # lint + auto-fix
uv run ruff format .               # format
uv run ruff format --check .       # format check
uv run ty check .                  # type check
```

## What Each Test Suite Covers

### Unit tests

Test isolated components and functions:
- **Provider functionality**: model initialization, API interactions, capability checks.
- **Tool operations**: all MCP tools (chat, analyze, debug, etc.).
- **Conversation memory**: threading, continuation, history management.
- **File handling**: path validation, token limits, deduplication.
- **Auto mode**: model selection logic and fallback behavior.

### HTTP recording/replay tests

Expensive API calls (like o3-pro) use custom recording/replay:
- **Real API validation**: tests against actual provider responses.
- **Cost efficiency**: record once, replay forever.
- **Provider compatibility**: validates fixes against real APIs.
- Uses HTTP Transport Recorder for httpx-based API calls.
- See [HTTP Recording/Replay Testing Guide](./vcr-testing.md) for details.

### Simulator tests

Validate real-world usage scenarios by simulating actual Claude prompts:
- **Basic conversations**: multi-turn chat functionality with real prompts.
- **Cross-tool continuation**: context preservation across different tools.
- **File deduplication**: efficient handling of repeated file references.
- **Model selection**: proper routing to configured providers.
- **Token allocation**: context window management in practice.
- **Persistence**: conversation persistence and retrieval.

## Contributing

For full contribution guidelines, testing requirements, and code quality standards, see the [Contributing Guide](./contributions.md).

### Quick testing reference

```bash
./code_quality_checks.sh                                  # quality gate
uv run pytest -xvs                                        # unit tests
uv run python communication_simulator_test.py --quick     # simulator smoke test
```

All tests must pass before submitting a PR. See the [Contributing Guide](./contributions.md) for complete requirements.
