#!/bin/bash
set -euo pipefail

# ============================================================================
# PAL MCP Server Setup Script
#
# uv-based dev bootstrap: syncs dependencies into .venv via uv, prepares .env,
# and optionally tails the log file.
#
# This does NOT register the server with an MCP client. The published package
# is the registration path — point your client at `uvx pal-mcp-server` and let
# it own the environment (see README). Self-registration used to hand-maintain
# a list of env keys to forward, which drifted from the provider registry.
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

readonly PLACEHOLDER_KEYS=(
    "GEMINI_API_KEY:your_gemini_api_key_here"
    "OPENAI_API_KEY:your_openai_api_key_here"
    "XAI_API_KEY:your_xai_api_key_here"
    "DIAL_API_KEY:your_dial_api_key_here"
    "OPENROUTER_API_KEY:your_openrouter_api_key_here"
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
  $0              Set up the dev environment (sync deps, prepare .env)
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
    # --locked, not --frozen: both refuse to re-lock, but --frozen also refuses
    # to *notice* a stale lock, so someone who added a dependency would get an
    # environment silently missing it and a ModuleNotFoundError later. --locked
    # errors here instead, matching code_quality_checks.sh's step 0.
    if ! uv sync --locked; then
        print_error "uv sync failed (if the lockfile is stale, run: uv lock)"
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

    echo ""
    print_success "Setup complete"
    echo "  Logs:      $(pwd)/$LOG_DIR/$LOG_FILE"
    echo "  Follow:    ./run-server.sh -f"
    echo "  Dev deps:  uv sync --group dev"
    echo ""
    echo "To use PAL from an MCP client, add this to its config:"
    echo ""
    echo '  "pal": { "command": "uvx", "args": ["pal-mcp-server"],'
    echo '           "env": { "GEMINI_API_KEY": "${GEMINI_API_KEY}" } }'
    echo ""
    echo "  To run this checkout instead of the published package:"
    echo "    uv run --project $(pwd) python server.py"
    echo ""

    if [[ $follow -eq 1 ]]; then
        follow_logs
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
