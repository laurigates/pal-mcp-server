"""The environment a relayed CLI receives is an allowlist, not a copy.

Regression cover for issue #119: `BaseCLIAgent._build_environment` used
`os.environ.copy()`, so relaying a prompt to one vendor's CLI handed that
process every other vendor's API key.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

import pytest

from clink.agents import create_agent
from clink.constants import PROJECT_ROOT
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.registry import ClinkRegistry

# The provider key each shipped CLI legitimately needs. `claude` authenticates
# with ANTHROPIC_API_KEY, which is not a PAL provider key and so never appears
# in the set derived from .env.example.
OWN_PROVIDER_KEY: dict[str, frozenset[str]] = {
    "claude": frozenset(),
    "codex": frozenset({"OPENAI_API_KEY"}),
    "gemini": frozenset({"GEMINI_API_KEY"}),
}


def _provider_api_keys() -> frozenset[str]:
    """Every provider API key PAL documents, read from .env.example.

    Derived rather than hardcoded so a newly documented provider is covered the
    moment it is added. Commented-out lines count: CUSTOM_API_KEY ships
    commented and is still a real credential in a configured deployment.
    """
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    return frozenset(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]*_API_KEY)\s*=", text, flags=re.MULTILINE))


def test_provider_key_set_is_not_vacuous():
    """Control for the leak assertions below: an empty set would pass them all."""
    keys = _provider_api_keys()
    assert {
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "DIAL_API_KEY",
        "OPENROUTER_API_KEY",
        "AZURE_OPENAI_API_KEY",
    } <= keys, f"parsed only {sorted(keys)} from .env.example"


@pytest.fixture()
def shipped_registry(monkeypatch, tmp_path):
    """Registry loaded from the repo's conf/cli_clients only.

    ~/.pal/cli_clients is always searched at runtime; pointing it at an empty
    directory keeps the assertions about a developer machine's overrides out.
    """
    monkeypatch.setattr("clink.registry.USER_CONFIG_DIR", tmp_path / "absent")
    monkeypatch.delenv("CLI_CLIENTS_CONFIG_PATH", raising=False)
    return ClinkRegistry()


@pytest.mark.parametrize("cli_name", sorted(OWN_PROVIDER_KEY))
def test_relayed_cli_receives_no_foreign_provider_key(monkeypatch, shipped_registry, cli_name):
    provider_keys = _provider_api_keys()
    for key in provider_keys:
        monkeypatch.setenv(key, f"secret-value-for-{key}")
    # Not a PAL provider key, but it is claude's own credential.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value-for-ANTHROPIC_API_KEY")

    agent = create_agent(shipped_registry.get_client(cli_name))
    env = agent._build_environment()

    own = OWN_PROVIDER_KEY[cli_name]
    leaked = (provider_keys & env.keys()) - own
    assert leaked == set(), f"{cli_name} was handed foreign provider keys: {sorted(leaked)}"

    # The mirror assertion: over-narrowing the allowlist would break the relay
    # at runtime while satisfying the leak check above.
    assert own <= env.keys(), f"{cli_name} lost its own credential"


def test_claude_keeps_its_own_vendor_credentials(monkeypatch, shipped_registry):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    env = create_agent(shipped_registry.get_client("claude"))._build_environment()

    assert env["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert env["ANTHROPIC_BASE_URL"] == "https://example.invalid"
    assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert "GEMINI_API_KEY" not in env


def _client(**overrides) -> ResolvedCLIClient:
    role = ResolvedCLIRole(
        name="default",
        prompt_path=Path("systemprompts/clink/default.txt").resolve(),
        role_args=[],
    )
    defaults = {
        "name": "gemini",
        "executable": ["gemini"],
        "internal_args": [],
        "config_args": [],
        "env": {},
        "env_passthrough": ["GEMINI_"],
        "timeout_seconds": 30,
        "parser": "gemini_json",
        "roles": {"default": role},
        "output_to_file": None,
        "working_dir": None,
    }
    defaults.update(overrides)
    return ResolvedCLIClient(**defaults)


def test_allowlist_passes_through_what_the_cli_needs(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")
    monkeypatch.setenv("HOME", "/home/relay-user")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.invalid:3128")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/etc/ssl/corp.pem")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/relay-user/.config")
    monkeypatch.setenv("SOME_UNRELATED_INTERNAL_VAR", "should-not-travel")

    env = create_agent(_client())._build_environment()

    assert env["PATH"] == "/usr/local/bin:/usr/bin"
    # HOME is a deliberate keep: claude/codex/gemini store their own
    # subscription credentials under it (issue #119, decision 2).
    assert env["HOME"] == "/home/relay-user"
    assert env["TERM"] == "xterm-256color"
    assert env["LANG"] == "en_US.UTF-8"
    assert env["LC_ALL"] == "en_US.UTF-8"
    assert env["HTTPS_PROXY"] == "http://proxy.example.invalid:3128"
    assert env["NODE_EXTRA_CA_CERTS"] == "/etc/ssl/corp.pem"
    assert env["XDG_CONFIG_HOME"] == "/home/relay-user/.config"
    assert "SOME_UNRELATED_INTERNAL_VAR" not in env


def test_client_env_block_is_still_honoured(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SOME_UNRELATED_INTERNAL_VAR", "should-not-travel")

    env = create_agent(_client(env={"CLINK_CUSTOM_FLAG": "on", "PATH": "/opt/bin"}))._build_environment()

    assert env["CLINK_CUSTOM_FLAG"] == "on"
    assert env["PATH"] == "/opt/bin", "the per-client env block must win over the allowlist"
    assert "SOME_UNRELATED_INTERNAL_VAR" not in env


class _DummyProcess:
    returncode = 0

    async def communicate(self, _input):
        return b'{"response": "ok"}', b""


async def _capture_cwd(monkeypatch, client) -> str:
    seen: dict[str, str] = {}

    async def fake_create_subprocess_exec(*_args, **kwargs):
        seen["cwd"] = kwargs["cwd"]
        return _DummyProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

    agent = create_agent(client)
    await agent.run(role=client.get_role("default"), prompt="hello", files=[], images=[])
    return seen["cwd"]


@pytest.mark.asyncio
async def test_unconfigured_working_dir_is_an_empty_scratch_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "secret-repo-file.txt").write_text("private", encoding="utf-8")

    cwd = await _capture_cwd(monkeypatch, _client(working_dir=None))

    assert cwd is not None, "the subprocess must not inherit the server's cwd"
    assert Path(cwd).resolve() != tmp_path.resolve()
    assert "clink-workspace-" in Path(cwd).name


@pytest.mark.asyncio
async def test_configured_working_dir_is_used(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    cwd = await _capture_cwd(monkeypatch, _client(working_dir=workspace))

    assert Path(cwd) == workspace


@pytest.mark.asyncio
async def test_scratch_directory_is_cleaned_up(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    cwd = await _capture_cwd(monkeypatch, _client(working_dir=None))

    assert not Path(cwd).exists()
