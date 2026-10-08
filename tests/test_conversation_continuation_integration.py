"""Integration test for conversation continuation persistence."""

from tools.chat import ChatRequest, ChatTool
from utils.conversation_memory import get_thread
from utils.storage_backend import get_storage_backend


def test_first_response_persisted_in_conversation_history(tmp_path):
    """Ensure the assistant's initial reply is stored for newly created threads."""

    # Clear in-memory storage to avoid cross-test contamination
    storage = get_storage_backend()
    storage._store.clear()  # type: ignore[attr-defined]

    tool = ChatTool()
    request = ChatRequest(
        prompt="First question?",
        model="local-llama",
        working_directory_absolute_path=str(tmp_path),
    )
    response_text = "Here is the initial answer."

    # Mimic the first tool invocation (no continuation_id supplied): execute()
    # opens the thread before the model call, then parses the reply into it.
    new_thread_id = tool._start_conversation_thread(request)
    output = tool._parse_response(
        response_text,
        request,
        {"model_name": "local-llama", "provider": "custom"},
        new_thread_id,
    )

    thread_id = output.continuation_offer.continuation_id
    assert thread_id == new_thread_id
    thread = get_thread(thread_id)

    assert thread is not None
    assert [turn.role for turn in thread.turns] == ["user", "assistant"]
    assert thread.turns[-1].content == response_text

    # Cleanup storage for subsequent tests
    storage._store.clear()  # type: ignore[attr-defined]
