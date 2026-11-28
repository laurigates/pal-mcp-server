"""Stage D prototype end to end: a 6-turn chat with files, through server.py and the strict stub.

With ``PAL_MESSAGE_HISTORY=true`` every request must be the previous request plus
the reply and the new input, message for message, which is the only shape GPT-5.6+
reads its cache from (tests/test_message_history.py). The control run without the
flag keeps the rendered single-message history.

Runs in a subprocess: the harness imports ``server`` with its own environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import asyncio, json, os, sys, tempfile
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "scripts"))
os.environ["PAL_MESSAGE_HISTORY"] = sys.argv[2]
import cache_prefix as cp
root = Path(tempfile.mkdtemp())
fixtures = root / "fixtures"; fixtures.mkdir()
cp.write_fixtures(fixtures)
h = cp.Harness(fixtures)
h.start_server(root / "state")
from providers.custom import CustomProvider
CustomProvider.SUPPORTS_MESSAGE_HISTORY = True  # the stub is served by the Custom provider
files = sorted(str(p) for p in fixtures.iterdir() if p.is_file())
requests = asyncio.run(h.chat_thread([f"question {i}" for i in range(1, 7)], {1: files[:1], 4: files[1:2]}))
print(json.dumps([r["messages"] for r in requests]))
"""


def _run(flag: str) -> list[list[dict]]:
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(REPO), flag], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _file_blocks(messages: list[dict]) -> int:
    return sum(str(m["content"]).count("--- BEGIN FILE:") for m in messages)


def test_each_request_appends_to_the_previous_one():
    requests = _run("true")
    for previous, current in zip(requests, requests[1:], strict=False):
        assert current[: len(previous)] == previous
        assert [m["role"] for m in current[len(previous) :]] == ["assistant", "user"]
    # Each file is sent once, in the message of the turn that attached it.
    assert _file_blocks(requests[-1]) == 2


def test_without_the_flag_history_stays_one_message():
    requests = _run("false")
    assert all(len(messages) == 2 for messages in requests)
