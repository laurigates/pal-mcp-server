"""Helpers for invoking the server's MCP handlers from tests.

The mcp 2.x SDK replaced the decorator handlers with constructor ``on_*``
parameters and changed their signatures: a handler now takes the request
context and a params object rather than loose arguments, and returns a
protocol-typed result rather than a bare list (issue #117).

Tests go through these helpers so that shape lives in one place. Calling
``handle_call_tool`` directly from a few dozen sites would mean every future
signature change is a sweep.
"""

from typing import Any

from mcp.types import CallToolRequestParams, CallToolResult, GetPromptRequestParams, PaginatedRequestParams


async def call_tool(name: str, arguments: dict[str, Any] | None = None, *, context: Any = None) -> CallToolResult:
    """Invoke ``tools/call`` the way the SDK does.

    ``context`` is None for a direct invocation, which is what the handler sees
    outside a live session; progress reporting is inert in that case.
    """
    from server import handle_call_tool

    return await handle_call_tool(context, CallToolRequestParams(name=name, arguments=arguments or {}))


async def call_tool_text(name: str, arguments: dict[str, Any] | None = None, *, context: Any = None) -> str:
    """Return the single text block of a successful ``tools/call``.

    Asserts the call did not come back as an error result, so a test that meant
    to exercise the success path fails on the error rather than on a confusing
    IndexError further down.
    """
    result = await call_tool(name, arguments, context=context)
    assert result.is_error is not True, f"tools/call for {name!r} returned an error result: {result.content}"
    assert len(result.content) == 1, f"expected one content block, got {len(result.content)}"
    return result.content[0].text


async def list_tools(*, context: Any = None):
    """Invoke ``tools/list`` and return the tools it advertises."""
    from server import handle_list_tools

    result = await handle_list_tools(context, PaginatedRequestParams())
    return result.tools


async def list_prompts(*, context: Any = None):
    """Invoke ``prompts/list`` and return the prompts it advertises."""
    from server import handle_list_prompts

    result = await handle_list_prompts(context, PaginatedRequestParams())
    return result.prompts


async def get_prompt(name: str, arguments: dict[str, Any] | None = None, *, context: Any = None):
    """Invoke ``prompts/get`` for a single prompt."""
    from server import handle_get_prompt

    return await handle_get_prompt(context, GetPromptRequestParams(name=name, arguments=arguments or {}))
