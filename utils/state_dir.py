"""Per-user state directory for PAL's on-disk artifacts (logs, transcripts)."""

from pathlib import Path

from utils.env import get_env


def get_state_dir() -> Path:
    """Return PAL's per-user state directory without creating it.

    Resolution order: ``PAL_STATE_DIR``, then ``$XDG_STATE_HOME/pal-mcp-server``,
    then ``~/.local/state/pal-mcp-server``.
    """
    override = get_env("PAL_STATE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_state_home = get_env("XDG_STATE_HOME")
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return base / "pal-mcp-server"
