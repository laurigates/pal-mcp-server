"""Tests for the conversation-scoped session identifier."""

import pytest

from utils.session_context import get_session_id, set_session_id


@pytest.fixture(autouse=True)
def _reset_session_id():
    yield
    set_session_id(None)


def test_falls_back_to_stable_process_id():
    set_session_id(None)
    first = get_session_id()
    assert first
    assert get_session_id() == first


def test_continuation_id_becomes_the_session_id():
    set_session_id("8f3c1f2e-0f3a-4b7a-9f4e-2b6a1c0d5e77")
    assert get_session_id() == "8f3c1f2e-0f3a-4b7a-9f4e-2b6a1c0d5e77"


def test_process_id_differs_from_conversation_id():
    set_session_id(None)
    process_id = get_session_id()
    set_session_id("thread-abc")
    assert get_session_id() != process_id


@pytest.mark.parametrize(
    "unsafe",
    [
        "thread\r\nx-injected: 1",
        "thread id with spaces",
        "ä" * 4,
        "x" * 129,
        "",
        123,
        None,
    ],
)
def test_unsafe_values_fall_back_to_the_process_id(unsafe):
    set_session_id(None)
    process_id = get_session_id()
    set_session_id(unsafe)
    assert get_session_id() == process_id
