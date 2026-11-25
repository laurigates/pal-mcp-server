"""`pal-mcp-server` command from the tombstone package: point at the new name, then run palaver."""

import sys

NOTICE = (
    "pal-mcp-server has been renamed to palaver-mcp-server. "
    "This command still works through a final tombstone release, but update your MCP config: "
    "`uvx pal-mcp-server` -> `uvx palaver-mcp-server` (keep `pal` as the server key). "
    "https://pypi.org/project/palaver-mcp-server/\n"
)


def main() -> None:
    # stderr only: stdout is the MCP stdio channel, and any byte written there
    # before the server starts breaks the client's protocol handshake.
    sys.stderr.write(NOTICE)
    sys.stderr.flush()

    from server import run  # provided by palaver-mcp-server

    run()


if __name__ == "__main__":
    main()
