"""Conversation threads exposed as MCP resources, tested through real request handling.

A handler-level test cannot catch the declared-vs-implemented gap: ``main()``
hand-builds ``ServerCapabilities``, so a server can register resource handlers yet
never advertise ``resources`` in its initialize response. The stdio test drives the
real ``python -m server`` entry point; the in-process tests go through the SDK's
request dispatch for the error and disabled paths.
"""

import json
import sys
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

# MCP spec, Resources > Error Handling: "Resource not found" is -32002. Written
# out rather than imported from server so a wrong constant there cannot pass.
RESOURCE_NOT_FOUND = -32002

THREAD_ID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
THREAD_URI = f"pal://threads/{THREAD_ID}"


def _write_fixture_transcript(state_dir: Path) -> Path:
    threads = state_dir / "threads"
    threads.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "thread",
            "thread_id": THREAD_ID,
            "parent_thread_id": None,
            "tool_name": "chat",
            "created_at": "2026-10-07T10:00:00+00:00",
        },
        {
            "type": "turn",
            "thread_id": THREAD_ID,
            "role": "user",
            "content": "Is the retry loop in fetch() bounded?",
            "timestamp": "2026-10-07T10:00:01+00:00",
            "files": ["/src/fetch.py"],
            "images": None,
            "tool_name": "chat",
            "model_provider": None,
            "model_name": None,
            "model_metadata": None,
        },
        {
            "type": "turn",
            "thread_id": THREAD_ID,
            "role": "assistant",
            "content": "No: it retries forever on 503.",
            "timestamp": "2026-10-07T10:00:04+00:00",
            "files": None,
            "images": None,
            "tool_name": "chat",
            "model_provider": "openai",
            "model_name": "gpt-5",
            "model_metadata": None,
        },
    ]
    path = threads / f"{THREAD_ID}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _text(result) -> str:
    assert len(result.contents) == 1
    return result.contents[0].text


@pytest.mark.asyncio
async def test_live_stdio_server_advertises_lists_and_reads_thread_resources(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    state_dir = tmp_path / "state"
    _write_fixture_transcript(state_dir)
    repo_root = Path(__file__).resolve().parent.parent
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server"],
        cwd=str(repo_root),
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": str(repo_root),
            "LOG_LEVEL": "ERROR",
            "OPENAI_API_KEY": "sk-test-not-used",
            "DEFAULT_MODEL": "auto",
            "PAL_STATE_DIR": str(state_dir),
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_resources()
            templates = await session.list_resource_templates()
            index = await session.read_resource("pal://threads")
            thread = await session.read_resource(THREAD_URI)
            with pytest.raises(MCPError) as bad_id:
                await session.read_resource("pal://threads/../../etc/passwd")

    assert init.capabilities.resources is not None, "server does not advertise the resources capability"

    uris = [str(r.uri) for r in listed.resources]
    assert uris == ["pal://threads", THREAD_URI]
    thread_resource = listed.resources[1]
    assert thread_resource.name.startswith("chat: Is the retry loop")
    assert "gpt-5 (openai)" in thread_resource.name
    assert thread_resource.mime_type == "text/markdown"

    assert [t.uri_template for t in templates.resource_templates] == ["pal://threads/{thread_id}"]

    index_md = _text(index)
    assert f"| `{THREAD_ID}` | chat | gpt-5 (openai) | 2 | 2026-10-07T10:00:04+00:00 |" in index_md

    thread_md = _text(thread)
    assert thread_md.startswith(f"# PAL thread `{THREAD_ID}`")
    assert "## Turn 2 · assistant · gpt-5 (openai) · 2026-10-07T10:00:04+00:00" in thread_md
    assert "No: it retries forever on 503." in thread_md
    assert "`/src/fetch.py`" in thread_md

    assert "Invalid thread id" in bad_id.value.message


@pytest.fixture
def pal_server():
    import server

    return server.server


@pytest.mark.asyncio
async def test_in_process_read_rejects_invalid_and_unknown_thread(pal_server, monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path))
    _write_fixture_transcript(tmp_path)
    async with Client(pal_server) as client:
        with pytest.raises(MCPError, match="Invalid thread id") as bad_id:
            await client.read_resource("pal://threads/not-a-uuid")
        with pytest.raises(MCPError, match="No transcript for thread") as missing:
            await client.read_resource("pal://threads/00000000-0000-4000-8000-000000000000")
        with pytest.raises(MCPError, match="Unknown resource") as unknown:
            await client.read_resource("pal://elsewhere")
        assert "No: it retries forever" in _text(await client.read_resource(THREAD_URI))
    assert bad_id.value.code == INVALID_PARAMS
    assert missing.value.code == RESOURCE_NOT_FOUND
    assert unknown.value.code == INVALID_PARAMS


@pytest.mark.asyncio
async def test_in_process_disabled_transcripts_give_empty_index(pal_server, monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PAL_TRANSCRIPTS", "false")
    _write_fixture_transcript(tmp_path)
    async with Client(pal_server) as client:
        listed = await client.list_resources()
        index = await client.read_resource("pal://threads")
        with pytest.raises(MCPError, match="PAL_TRANSCRIPTS") as disabled:
            await client.read_resource(THREAD_URI)
    assert disabled.value.code == RESOURCE_NOT_FOUND
    assert [str(r.uri) for r in listed.resources] == ["pal://threads"]
    assert "No conversation transcripts found" in _text(index)


@pytest.mark.asyncio
async def test_in_process_missing_directory_gives_empty_index(pal_server, monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "never-created"))
    async with Client(pal_server) as client:
        listed = await client.list_resources()
        index = await client.read_resource("pal://threads")
    assert [str(r.uri) for r in listed.resources] == ["pal://threads"]
    assert "No conversation transcripts found" in _text(index)
