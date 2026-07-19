"""Unit tests for the jules tool (Google Jules async coding agent)."""

import json

import httpx
import pytest

from tools.jules import JULES_API_BASE, JulesTool
from tools.shared.exceptions import ToolExecutionError


def _mock_tool(monkeypatch, handler) -> JulesTool:
    """Return a JulesTool whose HTTP client is backed by a MockTransport handler."""
    monkeypatch.setenv("JULES_API_KEY", "test-key")
    tool = JulesTool()

    def build_client(api_key: str) -> httpx.AsyncClient:
        assert api_key == "test-key"
        return httpx.AsyncClient(
            base_url=JULES_API_BASE,
            headers={"X-Goog-Api-Key": api_key, "Content-Type": "application/json"},
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(tool, "_build_client", build_client)
    return tool


def _payload(results):
    assert len(results) == 1
    return json.loads(results[0].text)


def test_schema_and_flags():
    tool = JulesTool()
    assert tool.get_name() == "jules"
    assert tool.requires_model() is False
    schema = tool.get_input_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["action"]
    assert set(schema["properties"]["action"]["enum"]) == {
        "list_sources",
        "create",
        "status",
        "message",
        "approve",
    }


@pytest.mark.asyncio
async def test_missing_api_key_errors(monkeypatch):
    monkeypatch.delenv("JULES_API_KEY", raising=False)
    tool = JulesTool()
    with pytest.raises(ToolExecutionError) as exc:
        await tool.execute({"action": "list_sources"})
    assert "JULES_API_KEY" in str(exc.value)


@pytest.mark.asyncio
async def test_list_sources(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/sources")
        assert request.headers["X-Goog-Api-Key"] == "test-key"
        return httpx.Response(
            200,
            json={"sources": [{"name": "sources/github/o/r", "id": "github/o/r", "githubRepo": {"owner": "o"}}]},
        )

    tool = _mock_tool(monkeypatch, handler)
    payload = _payload(await tool.execute({"action": "list_sources"}))
    assert payload["status"] == "success"
    sources = json.loads(payload["content"])
    assert sources[0]["name"] == "sources/github/o/r"


@pytest.mark.asyncio
async def test_create_defaults_to_auto_create_pr(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/sessions")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "abc123", "state": "QUEUED", "url": "https://jules/x"})

    tool = _mock_tool(monkeypatch, handler)
    payload = _payload(
        await tool.execute(
            {"action": "create", "prompt": "fix bug", "source": "sources/github/o/r", "starting_branch": "main"}
        )
    )
    assert payload["status"] == "success"
    body = captured["body"]
    assert body["prompt"] == "fix bug"
    assert body["automationMode"] == "AUTO_CREATE_PR"
    assert body["sourceContext"]["source"] == "sources/github/o/r"
    assert body["sourceContext"]["githubRepoContext"]["startingBranch"] == "main"
    assert "requirePlanApproval" not in body  # default False is omitted
    summary = json.loads(payload["content"])
    assert summary["session_id"] == "abc123"


@pytest.mark.asyncio
async def test_create_requires_prompt_and_source(monkeypatch):
    tool = _mock_tool(monkeypatch, lambda request: httpx.Response(200, json={}))
    with pytest.raises(ToolExecutionError):
        await tool.execute({"action": "create", "source": "sources/github/o/r"})
    with pytest.raises(ToolExecutionError):
        await tool.execute({"action": "create", "prompt": "do it"})


@pytest.mark.asyncio
async def test_status_merges_session_and_activities(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/activities"):
            return httpx.Response(
                200,
                json={
                    "activities": [
                        {"originator": "agent", "agentMessaged": {"agentMessage": "working"}},
                        {"originator": "system", "sessionCompleted": {}},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "abc123",
                "state": "COMPLETED",
                "outputs": [{"pullRequest": {"url": "https://github.com/o/r/pull/1"}}],
            },
        )

    tool = _mock_tool(monkeypatch, handler)
    payload = _payload(await tool.execute({"action": "status", "session_id": "sessions/abc123"}))
    result = json.loads(payload["content"])
    assert result["state"] == "COMPLETED"
    assert result["outputs"][0]["pullRequest"]["url"].endswith("/pull/1")
    kinds = {a["kind"] for a in result["activities"]}
    assert {"agent_message", "session_completed"} <= kinds


@pytest.mark.asyncio
async def test_message_and_approve_hit_custom_verbs(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    tool = _mock_tool(monkeypatch, handler)
    await tool.execute({"action": "message", "session_id": "abc", "prompt": "make it corgi themed"})
    await tool.execute({"action": "approve", "session_id": "abc"})
    assert any(p.endswith("/sessions/abc:sendMessage") for p in seen)
    assert any(p.endswith("/sessions/abc:approvePlan") for p in seen)


@pytest.mark.asyncio
async def test_api_error_surfaced(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="permission denied")

    tool = _mock_tool(monkeypatch, handler)
    with pytest.raises(ToolExecutionError) as exc:
        await tool.execute({"action": "list_sources"})
    assert "403" in str(exc.value)
