import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolRequestParams

from providers.registry import ModelProviderRegistry
from server import server as mcp_server


def _install_dummy_provider(monkeypatch):
    """Ensure preflight model checks succeed without real provider configuration."""

    class DummyProvider:
        def get_provider_type(self):
            return SimpleNamespace(value="dummy")

        def get_capabilities(self, model_name):
            return SimpleNamespace(
                supports_extended_thinking=False,
                allow_code_generation=False,
                supports_images=False,
                context_window=1_000_000,
                max_image_size_mb=10,
            )

    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_provider_for_model",
        classmethod(lambda cls, model_name: DummyProvider()),
    )
    monkeypatch.setattr(
        ModelProviderRegistry,
        "get_available_models",
        classmethod(lambda cls, respect_restrictions=False: {"gemini-2.5-flash": None}),
    )


@pytest.mark.asyncio
async def test_tool_execution_error_sets_is_error_flag_for_mcp_response(monkeypatch):
    """Ensure ToolExecutionError surfaces as CallToolResult with is_error=True."""

    _install_dummy_provider(monkeypatch)

    # v2 dispatches on the method string and exposes the registered entry through
    # get_request_handler; the request_handlers mapping keyed by request type is
    # gone. Reaching the handler this way still proves the SDK routes tools/call
    # to our boundary rather than asserting on an imported function.
    entry = mcp_server.get_request_handler("tools/call")
    assert entry is not None, "tools/call is not registered on the server"

    arguments = {
        "prompt": "Trigger working_directory_absolute_path validation failure",
        "working_directory_absolute_path": "relative/path",  # Not absolute -> ToolExecutionError from ChatTool
        "absolute_file_paths": [],
        "model": "gemini-2.5-flash",
    }

    result = await entry.handler(None, CallToolRequestParams(name="chat", arguments=arguments))

    assert result.is_error is True
    assert result.content, "Expected error response content"

    payload = result.content[0].text
    data = json.loads(payload)
    assert data["status"] == "error"
    assert "absolute" in data["content"].lower()
