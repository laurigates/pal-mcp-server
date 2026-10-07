"""Tests for utils.state_dir.get_state_dir."""

from utils.state_dir import get_state_dir


def test_pal_state_dir_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert get_state_dir() == tmp_path / "custom"


def test_xdg_state_home_used_when_no_override(monkeypatch, tmp_path):
    monkeypatch.delenv("PAL_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert get_state_dir() == tmp_path / "xdg" / "pal-mcp-server"


def test_default_is_home_local_state(monkeypatch, tmp_path):
    monkeypatch.delenv("PAL_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert get_state_dir() == tmp_path / ".local" / "state" / "pal-mcp-server"
