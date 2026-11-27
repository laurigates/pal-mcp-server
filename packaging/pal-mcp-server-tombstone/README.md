# pal-mcp-server has moved to palaver-mcp-server

This project is now published as
[**palaver-mcp-server**](https://pypi.org/project/palaver-mcp-server/)
([source](https://github.com/laurigates/palaver-mcp-server)). It is a divergent
continuation of
[BeehiveInnovations/pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server),
renamed so it no longer occupies the upstream project's name on PyPI.

`pal-mcp-server` 13.0.0 is the final release under this name. It installs
`palaver-mcp-server` and keeps the `pal-mcp-server` command working: the command
prints a notice on stderr and starts palaver.

## Update your MCP config

```diff
 "pal": {
   "command": "uvx",
-  "args": ["pal-mcp-server"]
+  "args": ["palaver-mcp-server"]
 }
```

Keep `pal` as the server key, so the tools stay `mcp__pal__*`. The `PAL_*`
environment variables and the `~/.local/state/pal-mcp-server` state directory
are unchanged.
