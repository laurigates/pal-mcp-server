# Logging

## Quick Start - Follow Logs

The easiest way to monitor logs is to use the `-f` flag when starting the server:

```bash
# Start server and automatically follow MCP logs
./run-server.sh -f
```

This will start the server and immediately begin tailing the MCP server logs, wherever they are configured to go.

## Log Files

Logs are stored in a per-user state directory, the same location whether the server runs from a checkout or through `uvx`:

| Setting | Log directory |
|---------|---------------|
| `PAL_LOG_DIR` set | `$PAL_LOG_DIR` |
| `PAL_STATE_DIR` set | `$PAL_STATE_DIR/logs` |
| `XDG_STATE_HOME` set | `$XDG_STATE_HOME/pal-mcp-server/logs` |
| none of the above (default) | `~/.local/state/pal-mcp-server/logs` |

The first row that applies wins. The server writes the resolved path at startup (`Logging to: .../mcp_server.log`). The Docker image sets `PAL_LOG_DIR=/app/logs`, so container logs stay on the `./logs` volume mount.

The directory holds:

- **`mcp_server.log`** - Main server operations, API calls, and errors
- **`mcp_activity.log`** - Tool calls and conversation tracking

Log files rotate automatically when they reach 20MB, keeping up to 10 rotated files.

## Viewing Logs

To monitor MCP server activity:

```bash
# Default location; substitute your PAL_LOG_DIR if you set one
LOG_DIR=~/.local/state/pal-mcp-server/logs

# Follow logs in real-time
tail -f $LOG_DIR/mcp_server.log

# View last 100 lines
tail -n 100 $LOG_DIR/mcp_server.log

# View activity logs (tool calls only)
tail -f $LOG_DIR/mcp_activity.log

# Search for specific patterns
grep "ERROR" $LOG_DIR/mcp_server.log
grep "tool_name" $LOG_DIR/mcp_activity.log
```

## Log Level

Set verbosity with `LOG_LEVEL` in your `.env` file:

```env
# Options: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

- **DEBUG**: Detailed information for debugging
- **INFO**: General operational messages (default)
- **WARNING**: Warning messages
- **ERROR**: Only error messages

## Log Format

Logs use a standardized format with timestamps:

```
2024-06-14 10:30:45,123 - module.name - INFO - Message here
```

## Tips

- Use `./run-server.sh -f` for the easiest log monitoring experience
- Activity logs show only tool-related events for cleaner output
- Main server logs include all operational details
- Logs persist across server restarts