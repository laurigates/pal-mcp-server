#!/bin/bash
set -euo pipefail

# ============================================================================
# PAL MCP Server Setup Script
#
# uv-based bootstrap: syncs dependencies into .venv via uv, prepares .env,
# registers the server with Claude Code, and optionally tails the log file.
#
# For development (linting, formatting, tests):
#     uv sync --group dev
#     ./code_quality_checks.sh
# ============================================================================

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

readonly LOG_DIR="logs"
readonly LOG_FILE="mcp_server.log"
readonly LEGACY_MCP_NAMES=("zen" "zen-mcp" "zen-mcp-server" "zen_mcp" "zen_mcp_server")

readonly PLACEHOLDER_KEYS=(
    "GEMINI_API_KEY:your_gemini_api_key_here"
    "OPENAI_API_KEY:your_openai_api_key_here"
    "XAI_API_KEY:your_xai_api_key_here"
    "DIAL_API_KEY:your_dial_api_key_here"
    "OPENROUTER_API_KEY:your_openrouter_api_key_here"
)

# Environment variables forwarded to `claude mcp add -e` if present and non-placeholder.
readonly FORWARDED_ENV_KEYS=(
    "GEMINI_API_KEY"
    "OPENAI_API_KEY"
    "XAI_API_KEY"
    "DIAL_API_KEY"
    "OPENROUTER_API_KEY"
    "CUSTOM_API_URL"
    "CUSTOM_API_KEY"
    "CUSTOM_MODEL_NAME"
    "DISABLED_TOOLS"
    "DEFAULT_MODEL"
    "LOG_LEVEL"
    "DEFAULT_THINKING_MODE_THINKDEEP"
    "CONVERSATION_TIMEOUT_HOURS"
    "MAX_CONVERSATION_TURNS"
)

# ----------------------------------------------------------------------------
# Utility output
# ----------------------------------------------------------------------------

print_success() { echo -e "${GREEN}✓${NC} $1" >&2; }
print_error()   { echo -e "${RED}✗${NC} $1" >&2; }
print_warning() { echo -e "${YELLOW}!${NC} $1" >&2; }
print_info()    { echo -e "${YELLOW}$1${NC}" >&2; }

# ----------------------------------------------------------------------------
# Help / version
# ----------------------------------------------------------------------------

get_version() {
    if [[ -f pyproject.toml ]]; then
        # Match the [project] version field; cheap and dependency-free.
        grep -E '^version[[:space:]]*=' pyproject.toml | head -1 \
            | sed -E 's/^version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/' \
            || echo "unknown"
    else
        echo "unknown"
    fi
}

show_help() {
    local version
    version=$(get_version)
    cat <<EOF
PAL MCP Server v${version}

Usage: $0 [OPTIONS]

Options:
  -h, --help      Show this help message
  -v, --version   Show version information
  -f, --follow    Set up, then follow server logs (tail -f $LOG_DIR/$LOG_FILE)

Examples:
  $0              Set up the MCP server (sync deps, prepare .env, register with Claude)
  $0 -f           Set up and follow logs in real-time

Development setup (linting, formatting, tests):
  uv sync --group dev
  ./code_quality_checks.sh

For more information:
  https://github.com/laurigates/pal-mcp-server
EOF
}

# ----------------------------------------------------------------------------
# Preflight: uv must be installed
# ----------------------------------------------------------------------------

require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        print_error "uv is not installed"
        echo "" >&2
        echo "Install uv with one of:" >&2
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        echo "  brew install uv" >&2
        echo "  pipx install uv" >&2
        echo "" >&2
        echo "Then re-run ./run-server.sh" >&2
        exit 1
    fi
}

# ----------------------------------------------------------------------------
# .env handling
# ----------------------------------------------------------------------------

setup_env_file() {
    if [[ -f .env ]]; then
        print_success ".env file already exists"
        return 0
    fi
    if [[ ! -f .env.example ]]; then
        print_error ".env.example not found — cannot create .env"
        return 1
    fi
    cp .env.example .env
    print_success "Created .env from .env.example"
}

# Warn (do not fail) if all keys are still placeholders.
check_api_keys() {
    local has_real_key=0
    local key_name placeholder current

    if [[ -f .env ]]; then
        # Source .env into a temporary scope so we don't pollute the shell.
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a
    fi

    for pair in "${PLACEHOLDER_KEYS[@]}"; do
        key_name="${pair%%:*}"
        placeholder="${pair##*:}"
        current="${!key_name:-}"
        if [[ -n "$current" && "$current" != "$placeholder" ]]; then
            print_success "$key_name configured"
            has_real_key=1
        fi
    done

    if [[ -n "${CUSTOM_API_URL:-}" ]]; then
        print_success "CUSTOM_API_URL configured: $CUSTOM_API_URL"
        has_real_key=1
    fi

    if [[ $has_real_key -eq 0 ]]; then
        print_warning "No real API keys detected in .env (all values look like placeholders)"
        echo "  Edit .env and set at least one of: GEMINI_API_KEY, OPENAI_API_KEY," >&2
        echo "  XAI_API_KEY, DIAL_API_KEY, OPENROUTER_API_KEY, or CUSTOM_API_URL." >&2
        echo "  Setup will continue; the server won't talk to any model until keys are set." >&2
    fi
}

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

sync_dependencies() {
    print_info "Syncing dependencies with uv..."
    # Runtime sync only — dev tools live in [dependency-groups.dev].
    if ! uv sync; then
        print_error "uv sync failed"
        return 1
    fi
    print_success "Dependencies installed into .venv"
}

# ----------------------------------------------------------------------------
# Log directory
# ----------------------------------------------------------------------------

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
    touch "$LOG_DIR/$LOG_FILE"
}

# ----------------------------------------------------------------------------
# Claude Code MCP registration
# ----------------------------------------------------------------------------

# Build "-e KEY=value" arguments for `claude mcp add` from a sourced .env.
collect_env_args() {
    local args=()
    local key value
    for key in "${FORWARDED_ENV_KEYS[@]}"; do
        value="${!key:-}"
        if [[ -z "$value" ]]; then
            continue
        fi
        # Skip placeholders like `your_gemini_api_key_here`.
        if [[ "$value" =~ ^your_.*_here$ ]]; then
            continue
        fi
        args+=(-e "${key}=${value}")
    done
    printf '%s\n' "${args[@]+"${args[@]}"}"
}

# Locate the `claude` binary, including common non-PATH locations.
locate_claude() {
    if command -v claude >/dev/null 2>&1; then
        return 0
    fi
    local candidate dir
    for candidate in "$HOME/.local/bin/claude" "/opt/homebrew/bin/claude" "/usr/local/bin/claude"; do
        if [[ -x "$candidate" ]]; then
            dir="$(dirname "$candidate")"
            export PATH="$dir:$PATH"
            print_info "Found Claude CLI at $candidate"
            return 0
        fi
    done
    return 1
}

register_with_claude() {
    if ! locate_claude; then
        print_warning "Claude CLI (\`claude\`) not found on PATH"
        echo "  Install Claude Code from https://docs.anthropic.com/en/docs/claude-code/cli-usage" >&2
        echo "  Then re-run ./run-server.sh to register the MCP server." >&2
        # Print the command they could run manually once installed.
        print_manual_registration
        return 0
    fi

    # Clean up legacy zen-named registrations from before the rename.
    local legacy
    for legacy in "${LEGACY_MCP_NAMES[@]}"; do
        claude mcp remove "$legacy" -s user >/dev/null 2>&1 || true
    done

    # Always re-register so the command/env stays current with the source tree.
    claude mcp remove pal -s user >/dev/null 2>&1 || true

    local server_path project_dir
    project_dir="$(pwd)"
    server_path="${project_dir}/server.py"

    # Read env args into an array.
    local env_args=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && env_args+=("$line")
    done < <(collect_env_args)

    # Use `uv run` so the registration is venv-independent and survives Python
    # bumps — uv picks up .python-version and the synced .venv automatically.
    if claude mcp add pal -s user "${env_args[@]+"${env_args[@]}"}" -- uv run --project "$project_dir" python "$server_path"; then
        print_success "Registered PAL with Claude Code (user scope)"
    else
        print_warning "Could not register PAL automatically. To add manually, run:"
        print_manual_registration
    fi
}

print_manual_registration() {
    local server_path project_dir
    project_dir="$(pwd)"
    server_path="${project_dir}/server.py"
    local env_args=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && env_args+=("$line")
    done < <(collect_env_args)
    echo "" >&2
    echo "  claude mcp add pal -s user ${env_args[*]+${env_args[*]}} -- uv run --project $project_dir python $server_path" >&2
    echo "" >&2
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

follow_logs() {
    echo ""
    echo "Following $LOG_DIR/$LOG_FILE (Ctrl+C to stop)..."
    echo ""
    tail -f "$LOG_DIR/$LOG_FILE"
}

main() {
    local follow=0
    case "${1:-}" in
        -h|--help)    show_help; exit 0 ;;
        -v|--version) get_version; exit 0 ;;
        -f|--follow)  follow=1 ;;
        "")           ;;
        *)
            print_error "Unknown option: $1"
            echo "" >&2
            show_help >&2
            exit 1
            ;;
    esac

    local header
    header="PAL MCP Server v$(get_version)"
    echo "$header"
    printf '%*s\n' "${#header}" '' | tr ' ' '='
    echo ""

    require_uv
    setup_env_file
    check_api_keys
    sync_dependencies
    ensure_log_dir
    register_with_claude

    echo ""
    print_success "Setup complete"
    echo "  Logs:      $(pwd)/$LOG_DIR/$LOG_FILE"
    echo "  Follow:    ./run-server.sh -f"
    echo "  Dev deps:  uv sync --group dev"
    echo ""

    if [[ $follow -eq 1 ]]; then
        follow_logs
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
