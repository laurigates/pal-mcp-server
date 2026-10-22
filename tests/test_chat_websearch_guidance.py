"""
Chat's web-search guidance reaches the provider request (issue #199).

`build_standard_prompt` placed the guidance between the system prompt and
`=== USER REQUEST ===`, and `prepare_chat_style_prompt` then kept only the text
after that marker, so the guidance was built on every call and discarded before
the request went out. It now travels in the user section, which leaves the
system prompt byte-stable across turns for prompt caching (docs/prompt-caching.md).
"""

import pytest

from tools.chat import ChatTool
from tools.shared.exceptions import ToolExecutionError

GUIDANCE_MARKER = "WEB SEARCH CAPABILITY"


async def _capture_chat_request(monkeypatch, tmp_path) -> dict:
    captured = {}

    async def _capture(self, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing the outgoing request")

    monkeypatch.setattr("providers.gemini.GeminiModelProvider.generate_content", _capture, raising=True)

    # The stub raises once the outgoing kwargs are captured; SimpleTool.execute
    # wraps any provider failure in ToolExecutionError.
    with pytest.raises(ToolExecutionError):
        await ChatTool().execute(
            {
                "prompt": "hello",
                "model": "gemini-2.5-flash",
                "working_directory_absolute_path": str(tmp_path),
            }
        )
    return captured


@pytest.mark.asyncio
async def test_guidance_is_in_the_provider_request_when_enabled(monkeypatch, tmp_path):
    captured = await _capture_chat_request(monkeypatch, tmp_path)

    assert GUIDANCE_MARKER in captured["prompt"], "chat's web-search guidance never reached the model"
    assert ChatTool().get_chat_style_websearch_guidance() in captured["prompt"]
    # The guidance rides in the user section; the system prompt stays the tool's own, unchanged.
    assert captured["system_prompt"] == ChatTool().get_system_prompt()
    assert GUIDANCE_MARKER not in captured["system_prompt"]


@pytest.mark.asyncio
async def test_no_guidance_section_when_instruction_is_empty(monkeypatch, tmp_path):
    # There is no setting that turns web-search guidance off; this covers the
    # empty-instruction path, where the user section must lead with the request.
    monkeypatch.setattr(ChatTool, "get_websearch_instruction", lambda self, tool_specific=None: "")

    captured = await _capture_chat_request(monkeypatch, tmp_path)

    assert GUIDANCE_MARKER not in captured["prompt"]
    assert GUIDANCE_MARKER not in captured["system_prompt"]
    assert captured["prompt"].startswith("=== USER REQUEST ===")
