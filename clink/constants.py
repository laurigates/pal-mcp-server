"""Internal defaults and constants for clink."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_STREAM_LIMIT = 10 * 1024 * 1024  # 10MB per stream

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_PROMPTS_DIR = PROJECT_ROOT / "systemprompts" / "clink"
CONFIG_DIR = PROJECT_ROOT / "conf" / "cli_clients"
USER_CONFIG_DIR = Path.home() / ".pal" / "cli_clients"


# ----------------------------------------------------------------------------
# Subprocess environment allowlist
# ----------------------------------------------------------------------------
# A relayed CLI is a third-party binary belonging to one vendor. Copying the
# server's environment into it hands that binary every *other* vendor's API key
# (see issue #119), so the environment is built by allowlist instead.
#
# The names below are the ones a CLI legitimately needs in order to run at all;
# none of them is a credential for a PAL provider. Each group is a deliberate
# decision, not an accident of inheritance:
#
#   * Process basics — without PATH the CLI cannot resolve the helpers it
#     shells out to. HOME is kept on purpose: `claude`, `codex` and `gemini`
#     store their own subscription/OAuth credentials under it, so a synthetic
#     HOME would break the authentication these relays depend on. The trade-off
#     is accepted knowingly: HOME also exposes other vendors' config to a
#     config-scanning CLI, and the per-vendor key split below is the lever we
#     do have.
#   * Identity and temp — several CLIs derive cache/state paths from USER and
#     TMPDIR; an unset TMPDIR silently relocates their scratch files.
#   * Terminal and locale — output formatting, colour, and text encoding.
#     Dropping LANG/LC_* makes non-ASCII prompts round-trip badly.
#   * Network and TLS — corporate proxies and custom CA bundles are the usual
#     reason a CLI cannot reach its own API at all.
#   * Node runtime — `claude` and `gemini` ship as Node programs, so
#     NODE_EXTRA_CA_CERTS/NODE_OPTIONS are part of "can it make a request".
#   * Windows equivalents — the runner already handles Windows explicitly
#     (see BaseCLIAgent.run); a subprocess there fails without SYSTEMROOT.
BASE_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Process basics
        "PATH",
        "HOME",
        "SHELL",
        # Identity and temp
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        # Terminal and locale
        "TERM",
        "COLORTERM",
        "TERM_PROGRAM",
        "NO_COLOR",
        "FORCE_COLOR",
        "CI",
        "LANG",
        "LANGUAGE",
        "TZ",
        # Network and TLS
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # Node runtime (claude and gemini are Node CLIs)
        "NODE_EXTRA_CA_CERTS",
        "NODE_OPTIONS",
        "NODE_PATH",
        # Windows
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "USERNAME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
    }
)

# Prefixes allowed for every client. LC_* completes the locale set; XDG_* is
# where Linux CLIs put config, cache and state, so dropping it relocates a
# CLI's own settings out from under it.
BASE_ENV_PREFIX_ALLOWLIST: tuple[str, ...] = ("LC_", "XDG_")

# Per-client credential passthrough: the prefixes of the variables belonging to
# *that* CLI's own vendor. Entries are matched as prefixes, so an exact name
# works too. This is what keeps `codex` from seeing GEMINI_API_KEY while still
# letting it authenticate with OPENAI_API_KEY.
CLIENT_ENV_PASSTHROUGH: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_", "CLAUDE_"),
    "codex": ("OPENAI_", "CODEX_"),
    "gemini": ("GEMINI_", "GOOGLE_"),
}


@dataclass(frozen=True)
class CLIInternalDefaults:
    """Internal defaults applied to a CLI client during registry load."""

    parser: str
    additional_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    env_passthrough: tuple[str, ...] = ()
    default_role_prompt: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    runner: str | None = None


INTERNAL_DEFAULTS: dict[str, CLIInternalDefaults] = {
    "gemini": CLIInternalDefaults(
        parser="gemini_json",
        additional_args=["-o", "json"],
        env_passthrough=CLIENT_ENV_PASSTHROUGH["gemini"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="gemini",
    ),
    "codex": CLIInternalDefaults(
        parser="codex_jsonl",
        additional_args=["exec"],
        env_passthrough=CLIENT_ENV_PASSTHROUGH["codex"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="codex",
    ),
    "claude": CLIInternalDefaults(
        parser="claude_json",
        additional_args=["--print", "--output-format", "json"],
        env_passthrough=CLIENT_ENV_PASSTHROUGH["claude"],
        default_role_prompt="systemprompts/clink/default.txt",
        runner="claude",
    ),
}
