"""The server stays usable with no provider API keys configured (issue #116).

Startup used to raise ``ValueError`` before ``stdio_server()`` came up, so a
first-run user with no key saw a subprocess that died rather than a server that
was running and unconfigured. These tests pin the replacement contract:

* ``configure_providers()`` records the condition instead of raising;
* ``tools/list`` still returns the full surface;
* a ``tools/call`` on a provider-requiring tool comes back as an MCP tool
  execution error result naming the environment variables to set;
* ``version`` and ``listmodels`` still run, so the state is diagnosable;
* any ``ToolExecutionError`` raised from a tool reaches the client through the
  explicit boundary in ``handle_call_tool`` rather than as an exception.
"""

import json

import pytest
from mcp.types import CallToolRequestParams, CallToolResult, TextContent

import server
import utils.model_restrictions as model_restrictions
from providers.registry import REGISTERED_PROVIDER_CLASSES, ModelProviderRegistry
from tests.mcp_call_helpers import call_tool, list_tools
from tools import VersionTool
from tools.shared.exceptions import ToolExecutionError


@pytest.fixture
def unconfigured_providers(monkeypatch):
    """Put the process in the genuine no-provider-configured state.

    ``tests/conftest.py``'s autouse ``_isolated_provider_registry`` fixture sets
    ``dummy-key-for-tests`` for the default provider trio and registers them, so
    a test that does not scrub them would assert against a *configured* server
    and pass vacuously. This clears every provider credential variable and every
    registered provider, then runs the real ``configure_providers()``.
    """
    for provider_cls in REGISTERED_PROVIDER_CLASSES:
        for var in provider_cls.credential_env_vars():
            monkeypatch.delenv(var, raising=False)

    registry = ModelProviderRegistry()
    registry._providers.clear()
    registry._initialized_providers.clear()
    model_restrictions._restriction_service = None

    monkeypatch.setattr(server, "_provider_configuration_error", None, raising=False)
    server.configure_providers()
    yield
    model_restrictions._restriction_service = None


def test_configure_providers_records_instead_of_raising(unconfigured_providers):
    """Step 1: the condition is recorded, not raised, so ``main()`` can serve."""
    message = server.get_provider_configuration_error()

    assert message is not None
    assert "At least one API configuration is required" in message
    # The remedy has to name the variables, or the client learns nothing useful.
    assert "GEMINI_API_KEY" in message
    assert "OPENAI_API_KEY" in message


@pytest.mark.asyncio
async def test_tools_list_is_unchanged_when_unconfigured(unconfigured_providers):
    """Step 2: the full tool surface stays discoverable."""
    tools = await list_tools()

    names = {tool.name for tool in tools}
    assert names == set(server.TOOLS.keys())
    assert {"version", "listmodels", "chat"} <= names


@pytest.mark.asyncio
async def test_provider_requiring_tool_returns_error_result(unconfigured_providers):
    """Step 3: a provider-requiring call is a tool execution error, not a crash."""
    result = await call_tool("chat", {"prompt": "hello", "model": "gemini-2.5-flash"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True

    payload = json.loads(result.content[0].text)
    assert payload["status"] == "error"
    assert "At least one API configuration is required" in payload["content"]
    assert "GEMINI_API_KEY" in payload["content"]
    assert payload["metadata"]["condition"] == "no_providers_configured"


@pytest.mark.asyncio
async def test_error_result_reaches_the_client_over_the_wire(unconfigured_providers):
    """The same call, reached through what the SDK will actually dispatch to.

    Going through the registered entry rather than importing the function keeps
    this honest: it fails if ``tools/call`` is ever wired to something else, or
    left unwired. v2 replaced the ``request_handlers`` mapping with
    ``get_request_handler(method)``, and dispatches on the method string.
    """
    entry = server.server.get_request_handler("tools/call")
    assert entry is not None, "tools/call is not registered on the server"
    assert entry.params_type is CallToolRequestParams

    params = CallToolRequestParams(
        name="chat",
        arguments={
            "prompt": "hello",
            "model": "gemini-2.5-flash",
            # Required by chat's input schema; the SDK validates before dispatch.
            "working_directory_absolute_path": "/tmp",
        },
    )

    result = await entry.handler(None, params)

    assert result.is_error is True
    assert "At least one API configuration is required" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["version", "listmodels"])
async def test_provider_free_tools_still_work_when_unconfigured(unconfigured_providers, tool_name):
    """Step 4: the diagnostic path stays open."""
    result = await call_tool(tool_name, {})

    # v2 returns a CallToolResult on the success path too, rather than a bare list.
    assert result.is_error is not True, f"{tool_name} came back as an error: {result.content}"
    assert result.content, f"{tool_name} returned no content"
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text.strip()


@pytest.mark.asyncio
async def test_tool_execution_error_is_converted_at_the_dispatch_boundary(monkeypatch):
    """A ``ToolExecutionError`` from a tool arrives as an error result.

    This is the contract the v2 SDK migration (#117) removes from the SDK: v2
    stops turning handler exceptions into ``CallToolResult(isError=True)``, so
    the conversion has to be ours. Asserting on ``handle_call_tool``'s own
    return value — rather than on the SDK handler around it — is what keeps this
    test meaningful once the SDK stops helping.
    """

    async def boom(self, arguments):
        raise ToolExecutionError('{"status": "error", "content": "deliberate failure"}')

    # Patched on the class rather than the registry instance, so it still applies
    # if the dispatcher instantiates the tool per call. ``version`` needs no
    # provider, which keeps this test about the boundary and nothing else.
    monkeypatch.setattr(VersionTool, "execute", boom)

    result = await call_tool("version", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    payload = json.loads(result.content[0].text)
    assert payload["content"] == "deliberate failure"


@pytest.mark.asyncio
async def test_non_tool_execution_exceptions_also_become_error_results(monkeypatch):
    """An exception that is *not* a ToolExecutionError still reaches the client.

    The 1.x SDK wrapped every handler exception, so code outside the 16
    ``raise ToolExecutionError`` sites relied on it without saying so. v2 wraps
    nothing, and a bare exception becomes a JSON-RPC protocol error whose
    message the client cannot act on. Without the boundary's general clause this
    returns a protocol error instead of a result.
    """

    async def boom(self, arguments):
        raise RuntimeError("something nobody wrapped")

    monkeypatch.setattr(VersionTool, "execute", boom)

    result = await call_tool("version", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "something nobody wrapped" in result.content[0].text


@pytest.mark.asyncio
async def test_expired_continuation_id_is_reported_as_a_result_not_a_protocol_error():
    """The concrete case that regressed: a stale continuation_id.

    ``reconstruct_thread_context`` raises a plain ``ValueError`` telling the
    caller to start a new thread. That message is a documented recovery path, so
    it has to arrive as tool output the agent can read — the cross-tool
    simulator scenario fails otherwise.
    """
    result = await call_tool(
        "version",
        {"continuation_id": "00000000-0000-4000-8000-000000000000"},
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    text = result.content[0].text
    assert "was not found or has expired" in text
    assert "continuation_id" in text


@pytest.mark.asyncio
async def test_successful_call_is_not_marked_as_an_error(unconfigured_providers):
    """The boundary must not turn ordinary results into error results.

    Under 1.x this could be asserted by type — the success path returned a bare
    list and only the error path built a ``CallToolResult``. v2 returns a
    ``CallToolResult`` either way, so the flag is what separates them, and it is
    what the client reads.
    """
    result = await call_tool("version", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is not True
    assert result.content and result.content[0].text.strip()
