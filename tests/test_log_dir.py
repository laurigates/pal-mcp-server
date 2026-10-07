"""Tests for utils.log_dir.get_log_dir."""

from utils.log_dir import get_log_dir
from utils.state_dir import get_state_dir


def test_pal_log_dir_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("PAL_LOG_DIR", str(tmp_path / "logs-here"))
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    assert get_log_dir() == tmp_path / "logs-here"


def test_pal_log_dir_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PAL_LOG_DIR", "~/pal-logs")
    assert get_log_dir() == tmp_path / "pal-logs"


def test_defaults_to_state_dir_logs(monkeypatch, tmp_path):
    monkeypatch.delenv("PAL_LOG_DIR", raising=False)
    monkeypatch.setenv("PAL_STATE_DIR", str(tmp_path / "state"))
    assert get_log_dir() == get_state_dir() / "logs"
    assert get_log_dir() == tmp_path / "state" / "logs"
