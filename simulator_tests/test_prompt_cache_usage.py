#!/usr/bin/env python3
"""Prompt-cache usage reaches the transcript (issue #193, stage 0).

Runs a four-turn chat thread and reads the thread's transcript from the state
directory the server writes it to. Every assistant turn must carry the usage
the provider reported.

Against the strict stub (``simulator_tests/stub_provider.py``) the check is
exact about the cache field: the stub reports ``cached_tokens`` as the prefix a
request shares with the previous request of the same thread, so turns 1 and 2
(no earlier request of the thread carries its ``Thread:`` header) must record
0 and turns 3 and 4 a positive count. That pins the whole path: the stub's
``prompt_tokens_details.cached_tokens``, the Chat Completions parser's
``cached_input_tokens``, and the transcript writer. Against a real server the
field depends on the server, so only the presence of usage is checked.
"""

from __future__ import annotations

import json
import os
import urllib.request

from utils.transcripts import transcript_path

from .base_test import BaseSimulatorTest

#: ``owned_by`` the stub reports on /v1/models; how this scenario knows it is talking to it.
STUB_OWNER = "simulator-stub"
TURNS = 4


class PromptCacheUsageTest(BaseSimulatorTest):
    """Usage, including the provider's cached-token count, is recorded per turn."""

    provider_agnostic = True

    @property
    def test_name(self) -> str:
        return "prompt_cache_usage"

    @property
    def test_description(self) -> str:
        return "Provider cached-token usage reaches the transcript"

    def _serving_stub(self) -> bool:
        base_url = (os.environ.get("CUSTOM_API_URL") or "").rstrip("/")
        if not base_url:
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                models = json.loads(response.read()).get("data", [])
        except Exception:
            return False
        return any(entry.get("owned_by") == STUB_OWNER for entry in models)

    def _assistant_usages(self, thread_id: str) -> list[dict]:
        path = transcript_path(thread_id)
        if not path.exists():
            self.logger.error(f"No transcript at {path}")
            return []
        usages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("type") == "turn" and record.get("role") == "assistant" and record.get("tool_name") == "chat":
                usages.append((record.get("model_metadata") or {}).get("usage") or {})
        return usages

    def run_test(self) -> bool:
        try:
            self.logger.info("Test: prompt-cache usage in the transcript")
            continuation_id = None
            for turn in range(1, TURNS + 1):
                params = {"prompt": f"Turn {turn}: say something short.", "model": self.simulator_model}
                if continuation_id:
                    params["continuation_id"] = continuation_id
                response, new_id = self.call_mcp_tool("chat", params)
                if not response:
                    self.logger.error(f"Turn {turn} returned no response")
                    return False
                continuation_id = continuation_id or new_id
                if not continuation_id:
                    self.logger.error("Turn 1 returned no continuation_id")
                    return False

            usages = self._assistant_usages(continuation_id)
            if len(usages) != TURNS:
                self.logger.error(f"Expected {TURNS} assistant chat turns in the transcript, found {len(usages)}")
                return False
            if not all(isinstance(u.get("input_tokens"), int) and u["input_tokens"] > 0 for u in usages):
                self.logger.error(f"A turn recorded no input tokens: {usages}")
                return False

            if not self._serving_stub():
                self.logger.info("  Not the stub provider: cached-token values are server-specific, not checked")
                return True

            cached = [u.get("cached_input_tokens") for u in usages]
            self.logger.info(f"  cached_input_tokens per turn: {cached}")
            if cached[:2] != [0, 0]:
                self.logger.error(f"Turns 1-2 must record a miss (0), got {cached[:2]}")
                return False
            if not all(isinstance(c, int) and c > 0 for c in cached[2:]):
                self.logger.error(f"Turns 3+ must record the shared prefix as cached, got {cached[2:]}")
                return False
            if any(c > u["input_tokens"] for c, u in zip(cached, usages, strict=True)):
                self.logger.error(f"A cached count exceeds its input tokens: {usages}")
                return False

            self.logger.info("  ✅ Cached-token usage reaches the transcript")
            return True
        except Exception as e:
            self.logger.error(f"Prompt-cache usage test failed: {e}")
            return False
