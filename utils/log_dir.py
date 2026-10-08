"""Location of PAL's log files (``mcp_server.log``, ``mcp_activity.log``)."""

from pathlib import Path

from utils.env import get_env
from utils.state_dir import get_state_dir


def get_log_dir() -> Path:
    """Return the directory the server writes its log files to, without creating it.

    ``PAL_LOG_DIR`` wins when set; otherwise ``<state dir>/logs`` (see
    :func:`utils.state_dir.get_state_dir`). The server, ``run-server.sh -f``
    and the simulator log readers all resolve through this function, so they
    agree on the location for a given environment.
    """
    override = get_env("PAL_LOG_DIR")
    if override:
        return Path(override).expanduser()
    return get_state_dir() / "logs"
