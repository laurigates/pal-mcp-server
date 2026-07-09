# PAL MCP: Many Workflows. One Context.

A Model Context Protocol server that connects AI CLIs to multiple AI models. It works with Claude Code, Gemini CLI, Codex CLI, Qwen Code CLI, and Cursor, and supports providers including Gemini, OpenAI, Anthropic, Grok, Azure, Ollama, OpenRouter, DIAL, and OpenCode Go.

---

## CLI-to-CLI Bridge (clink)

The [`clink`](docs/tools/clink.md) (CLI + Link) tool bridges external AI CLIs such as Gemini CLI, Codex CLI, and Claude Code into a single workflow. It launches isolated CLI subagents from within your current session — for example, Claude Code can spawn a Codex subagent for an isolated code review — so heavy investigations run in a fresh context while your main session stays unpolluted, with each subagent returning only its final result. Subagents can take specialized roles (`planner`, `codereviewer`, or custom) and participate in conversation continuity across tools. See [`docs/tools/clink.md`](docs/tools/clink.md) for details.

---

## Why PAL MCP?

PAL MCP is a Model Context Protocol server that connects AI CLIs like [Claude Code](https://www.anthropic.com/claude-code), [Codex CLI](https://developers.openai.com/codex/cli), and IDE clients such as [Cursor](https://cursor.com) to multiple AI models, enabling enhanced code analysis, problem-solving, and collaborative development within a single conversation.

It supports **conversation threading**, so your CLI can discuss ideas with multiple AI models, exchange reasoning, get second opinions, and run collaborative debates between models to reach deeper insights. Context carries forward seamlessly across tools and models, enabling workflows like multi-model code review → automated planning → implementation → pre-commit validation within a single thread — one model remembers what another said steps earlier.

You stay in control. Your CLI orchestrates the AI team, but you decide the workflow, crafting prompts that bring in Gemini Pro, GPT-5, Flash, or local offline models exactly when needed.

## Quick Start (5 minutes)

**Prerequisites:** Python 3.10+, Git, [uv installed](https://docs.astral.sh/uv/getting-started/installation/)

**1. Get API Keys** (choose one or more):
- [OpenRouter](https://openrouter.ai/) - Access multiple models with one API
- [Gemini](https://makersuite.google.com/app/apikey) - Google's latest models
- [OpenAI](https://platform.openai.com/api-keys) - O3, GPT-5 series
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) - Enterprise deployments of GPT-4o, GPT-4.1, GPT-5 family
- [X.AI](https://console.x.ai/) - Grok models
- [DIAL](https://dialx.ai/) - Vendor-agnostic model access
- [OpenCode Go](https://opencode.ai/docs/go/) - Flat-rate subscription for open-source coding models
- [Ollama](https://ollama.ai/) - Local models (free)

**2. Install** (choose one):

**Option A: Clone and Automatic Setup** (recommended)
```bash
git clone https://github.com/BeehiveInnovations/pal-mcp-server.git
cd pal-mcp-server

# Handles everything: setup, config, API keys from system environment.
# Auto-configures Claude Desktop, Claude Code, Gemini CLI, Codex CLI, Qwen CLI
# Enable / disable additional settings in .env
./run-server.sh
```

**Option B: Instant Setup with [uvx](https://docs.astral.sh/uv/getting-started/installation/)**

Only set the keys for providers you want to use; at least one is required.
```json
// Add to ~/.claude/settings.json or .mcp.json
{
  "mcpServers": {
    "pal": {
      "command": "bash",
      "args": ["-c", "for p in $(which uvx 2>/dev/null) $HOME/.local/bin/uvx /opt/homebrew/bin/uvx /usr/local/bin/uvx uvx; do [ -x \"$p\" ] && exec \"$p\" --from git+https://github.com/BeehiveInnovations/pal-mcp-server.git pal-mcp-server; done; echo 'uvx not found' >&2; exit 1"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:~/.local/bin",
        "GEMINI_API_KEY": "your-gemini-key",
        "OPENAI_API_KEY": "your-openai-key",
        "AZURE_OPENAI_API_KEY": "your-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/",
        "XAI_API_KEY": "your-xai-key",
        "DIAL_API_KEY": "your-dial-key",
        "OPENCODE_API_KEY": "your-opencode-key",
        "OPENROUTER_API_KEY": "your-openrouter-key",
        "CUSTOM_API_URL": "http://localhost:11434/v1",
        "DISABLED_TOOLS": "analyze,refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "auto"
      }
    }
  }
}
```

**3. Start Using!**
```
"Use pal to analyze this code for security issues with gemini pro"
"Debug this error with o3 and then get flash to suggest optimizations"
"Plan the migration strategy with pal, get consensus from multiple models"
"clink with cli_name=\"gemini\" role=\"planner\" to draft a phased rollout plan"
```

- [Complete Setup Guide](docs/getting-started.md) with detailed installation, configuration for Gemini / Codex / Qwen, and troubleshooting
- [Cursor & VS Code Setup](docs/getting-started.md#ide-clients) for IDE integration instructions
- [Watch tools in action](#watch-tools-in-action) to see real-world examples

## Provider Configuration

PAL activates any provider that has credentials in your `.env`. Set the environment variables for the providers you want to use; at least one is required.

**Provider variables**

| Provider | Required | Optional |
|---|---|---|
| Gemini | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| OpenAI | `OPENAI_API_KEY` | — |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ALLOWED_MODELS`, `AZURE_MODELS_CONFIG_PATH` |
| X.AI | `XAI_API_KEY` | — |
| DIAL | `DIAL_API_KEY` | `DIAL_API_HOST`, `DIAL_API_VERSION` |
| OpenCode Go | `OPENCODE_API_KEY` | `OPENCODE_GO_ALLOWED_MODELS`, `OPENCODE_GO_MODELS_CONFIG_PATH` |
| OpenRouter | `OPENROUTER_API_KEY` | — |
| Local/Custom (Ollama, vLLM, LM Studio) | `CUSTOM_API_URL`, `CUSTOM_API_KEY`, `CUSTOM_MODEL_NAME` | — |

**General configuration**

| Variable | Purpose |
|---|---|
| `DEFAULT_MODEL` | Default model (`auto` lets the CLI pick per task) |
| `DEFAULT_THINKING_MODE_THINKDEEP` | Reasoning depth for `thinkdeep` (`minimal`/`low`/`medium`/`high`/`max`) |
| `DISABLED_TOOLS` | Comma-separated list of tools to disable |
| `LOG_LEVEL` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |
| `CONVERSATION_TIMEOUT_HOURS` | How long AI-to-AI threads persist before expiring |
| `MAX_CONVERSATION_TURNS` | Maximum turns in an AI-to-AI conversation thread |
| `LOCALE` | Response language (e.g. `fr-FR`, `en-US`) |
| `PAL_MCP_FORCE_ENV_OVERRIDE` | When `true`, `.env` values override system environment variables |

See [`.env.example`](.env.example) for the full annotated reference and [Configuration](docs/configuration.md) for details.

## Core Tools

> **Note:** Each tool ships with its own multi-step workflow, parameters, and descriptions that consume context window space even when unused. To optimize performance, some tools are disabled by default — see [Tool Configuration](#tool-configuration) to enable them.

**Collaboration & Planning** *(Enabled by default)*
- [`clink`](docs/tools/clink.md) - Bridge requests to external AI CLIs (Gemini planner, codereviewer, etc.)
- [`chat`](docs/tools/chat.md) - Brainstorm ideas, get second opinions, validate approaches. With capable models (GPT-5.2 Pro, Gemini 3.0 Pro), generates complete code / implementation
- [`thinkdeep`](docs/tools/thinkdeep.md) - Extended reasoning, edge case analysis, alternative perspectives
- [`planner`](docs/tools/planner.md) - Break down complex projects into structured, actionable plans
- [`consensus`](docs/tools/consensus.md) - Get expert opinions from multiple AI models with stance steering

**Code Analysis & Quality**
- [`debug`](docs/tools/debug.md) - Systematic investigation and root cause analysis
- [`precommit`](docs/tools/precommit.md) - Validate changes before committing, prevent regressions
- [`codereview`](docs/tools/codereview.md) - Professional reviews with severity levels and actionable feedback
- [`analyze`](docs/tools/analyze.md) *(disabled by default)* - Understand architecture, patterns, dependencies across entire codebases

**Development Tools** *(Disabled by default)*
- [`refactor`](docs/tools/refactor.md) - Intelligent code refactoring with decomposition focus
- [`testgen`](docs/tools/testgen.md) - Comprehensive test generation with edge cases
- [`secaudit`](docs/tools/secaudit.md) - Security audits with OWASP Top 10 analysis
- [`docgen`](docs/tools/docgen.md) - Generate documentation with complexity analysis

**Utilities**
- [`apilookup`](docs/tools/apilookup.md) - Forces current-year API/SDK documentation lookups in a sub-process (saves tokens within the current context window), prevents outdated training data responses
- [`challenge`](docs/tools/challenge.md) - Prevent "You're absolutely right!" responses with critical analysis
- [`tracer`](docs/tools/tracer.md) *(disabled by default)* - Static analysis prompts for call-flow mapping

## Tool Configuration

<details>
<summary><b id="tool-configuration">Tool Configuration</b></summary>

### Default Configuration

To optimize context window usage, only essential tools are enabled by default.

**Enabled by default:** `chat`, `thinkdeep`, `planner`, `consensus`, `codereview`, `precommit`, `debug`, `apilookup`, `challenge`

**Disabled by default:** `analyze`, `refactor`, `testgen`, `secaudit`, `docgen`, `tracer`

### Enabling Additional Tools

Remove a tool from the `DISABLED_TOOLS` list to enable it.

**Option 1: Edit your `.env` file**
```bash
# Default (from .env.example)
DISABLED_TOOLS=analyze,refactor,testgen,secaudit,docgen,tracer

# Enable analyze: remove it from the list
DISABLED_TOOLS=refactor,testgen,secaudit,docgen,tracer

# Enable all tools
DISABLED_TOOLS=
```

**Option 2: Configure in MCP settings**
```json
{
  "mcpServers": {
    "pal": {
      "env": {
        "DISABLED_TOOLS": "refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "pro",
        "DEFAULT_THINKING_MODE_THINKDEEP": "high",
        "GEMINI_API_KEY": "your-gemini-key",
        "OPENAI_API_KEY": "your-openai-key",
        "OPENROUTER_API_KEY": "your-openrouter-key",
        "LOG_LEVEL": "INFO",
        "CONVERSATION_TIMEOUT_HOURS": "6",
        "MAX_CONVERSATION_TURNS": "50"
      }
    }
  }
}
```

**Option 3: Enable all tools**
```json
{
  "mcpServers": {
    "pal": {
      "env": {
        "DISABLED_TOOLS": ""
      }
    }
  }
}
```

**Note:** Essential tools (`version`, `listmodels`) cannot be disabled. Restart your CLI session after changing tool configuration for changes to take effect. Each enabled tool adds to context window usage, so only enable what you need.

</details>

## Watch Tools In Action

Videos demonstrating the `chat`, `consensus`, `precommit`, `apilookup`, and `challenge` tools are available — covering collaborative decision making, multi-model debate, pre-commit validation, current-year API lookups, and critical analysis prompts.

## Key Features

**AI Orchestration**
- Auto model selection — the CLI picks the right AI for each task
- Multi-model workflows — chain different models in single conversations
- Conversation continuity — context preserved across tools and models
- [Context revival](docs/context-revival.md) — continue conversations even after context resets

**Model Support**
- Multiple providers — Gemini, OpenAI, Azure, X.AI, OpenRouter, DIAL, OpenCode Go, Ollama
- Latest models — GPT-5, Gemini 3.0 Pro, O3, Grok-4, local Llama
- [Thinking modes](docs/advanced-usage.md#thinking-modes) — control reasoning depth vs cost
- Vision support — analyze images, diagrams, screenshots

**Developer Experience**
- Guided workflows — systematic investigation prevents rushed analysis
- Smart file handling — auto-expand directories, manage token limits
- Web search integration — access current documentation and best practices
- [Large prompt support](docs/advanced-usage.md#working-with-large-prompts) — bypass MCP's 25K token limit

## Example Workflows

**Multi-model Code Review:**
```
"Perform a codereview using gemini pro and o3, then use planner to create a fix strategy"
```
→ Claude reviews code systematically → Consults Gemini Pro → Gets O3's perspective → Creates unified action plan

**Collaborative Debugging:**
```
"Debug this race condition with max thinking mode, then validate the fix with precommit"
```
→ Deep investigation → Expert analysis → Solution implementation → Pre-commit validation

See the [Advanced Usage Guide](docs/advanced-usage.md) for complex workflows, model configuration, and power-user features.

## Quick Links

### Documentation
- [Docs Overview](docs/index.md) - High-level map of major guides
- [Architecture](docs/architecture.md) - How the major subsystems fit together (new contributors start here)
- [Getting Started](docs/getting-started.md) - Complete setup guide
- [Tools Reference](docs/tools/) - All tools with examples
- [Advanced Usage](docs/advanced-usage.md) - Power user features
- [Configuration](docs/configuration.md) - Environment variables, restrictions
- [Adding Providers](docs/adding_providers.md) - Provider-specific setup (OpenAI, Azure, custom gateways)
- [Model Ranking Guide](docs/model_ranking.md) - How intelligence scores drive auto-mode suggestions

### Setup & Support
- [WSL Setup](docs/wsl-setup.md) - Windows users
- [Troubleshooting](docs/troubleshooting.md) - Common issues
- [Contributing](docs/contributions.md) - Code standards, PR process

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MCP (Model Context Protocol)](https://modelcontextprotocol.com)
- [Codex CLI](https://developers.openai.com/codex/cli)
- [Claude Code](https://claude.ai/code)
- [Gemini](https://ai.google.dev/)
- [OpenAI](https://openai.com/)
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
