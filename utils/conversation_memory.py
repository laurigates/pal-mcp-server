"""
Conversation Memory for AI-to-AI Multi-turn Discussions

This module provides conversation persistence and context reconstruction for
stateless MCP (Model Context Protocol) environments. It enables multi-turn
conversations between the agent and downstream models by storing conversation
state in memory across independent request cycles.

CRITICAL ARCHITECTURAL REQUIREMENT:
This conversation memory system is designed for PERSISTENT MCP SERVER PROCESSES.
It uses in-memory storage that persists only within a single Python process.

⚠️  IMPORTANT: This system will NOT work correctly if MCP tool calls are made
    as separate subprocess invocations (each subprocess starts with empty memory).

    WORKING SCENARIO: Claude Desktop with persistent MCP server process
    FAILING SCENARIO: Simulator tests calling server.py as individual subprocesses

    Root cause of test failures: Each subprocess call loses the conversation
    state from previous calls because memory is process-specific, not shared
    across subprocess boundaries.

ARCHITECTURE OVERVIEW:
The MCP protocol is inherently stateless - each tool request is independent
with no memory of previous interactions. This module bridges that gap by:

1. Creating persistent conversation threads with unique UUIDs
2. Storing complete conversation context (turns, files, metadata) in memory
3. Reconstructing conversation history when tools are called with continuation_id
4. Supporting cross-tool continuation - seamlessly switch between different tools
   while maintaining full conversation context and file references

CROSS-TOOL CONTINUATION:
A conversation started with one tool (e.g., 'analyze') can be continued with
any other tool (e.g., 'codereview', 'debug', 'chat') using the same continuation_id.
The second tool will have access to:
- All previous conversation turns and responses
- File context from previous tools (preserved in conversation history)
- Original thread metadata and timing information
- Accumulated knowledge from the entire conversation

Key Features:
- UUID-based conversation thread identification with security validation
- Turn-by-turn conversation history storage with tool attribution
- Cross-tool continuation support - switch tools while preserving context
- File context preservation - files shared in earlier turns remain accessible
- NEWEST-FIRST FILE PRIORITIZATION - when the same file appears in multiple turns,
  references from newer turns take precedence over older ones. This ensures the
  most recent file context is preserved when token limits require exclusions.
- Automatic turn limiting (20 turns max) to prevent runaway conversations
- Context reconstruction for stateless request continuity
- In-memory persistence with automatic expiration (3 hour TTL)
- Thread-safe operations for concurrent access
- Graceful degradation when storage is unavailable

DUAL PRIORITIZATION STRATEGY (Files & Conversations):
The conversation memory system implements sophisticated prioritization for both files and
conversation turns, using a consistent "newest-first" approach during collection but
presenting information in the optimal format for LLM consumption:

FILE PRIORITIZATION (Newest-First Throughout):
1. When collecting files across conversation turns, the system walks BACKWARDS through
   turns (newest to oldest) and builds a unique file list
2. If the same file path appears in multiple turns, only the reference from the
   NEWEST turn is kept in the final list
3. This "newest-first" ordering is preserved throughout the entire pipeline:
   - get_conversation_file_list() establishes the order
   - build_conversation_history() maintains it during token budgeting
   - When token limits are hit, OLDER files are excluded first
4. This strategy works across conversation chains - files from newer turns in ANY
   thread take precedence over files from older turns in ANY thread

CONVERSATION TURN PRIORITIZATION (Newest-First Collection, Chronological Presentation):
1. COLLECTION PHASE: Processes turns newest-to-oldest to prioritize recent context
   - When token budget is tight, OLDER turns are excluded first
   - Ensures most contextually relevant recent exchanges are preserved
2. PRESENTATION PHASE: Reverses collected turns to chronological order (oldest-first)
   - LLM sees natural conversation flow: "Turn 1 → Turn 2 → Turn 3..."
   - Maintains proper sequential understanding while preserving recency prioritization

This dual approach ensures optimal context preservation (newest-first) with natural
conversation flow (chronological) for maximum LLM comprehension and relevance.

USAGE EXAMPLE:
1. Tool A creates thread: create_thread("analyze", request_data) → returns UUID
2. Tool A adds response: add_turn(UUID, "assistant", response, files=[...], tool_name="analyze")
3. Tool B continues thread: get_thread(UUID) → retrieves full context
4. Tool B sees conversation history via build_conversation_history()
5. Tool B adds its response: add_turn(UUID, "assistant", response, tool_name="codereview")

DUAL STRATEGY EXAMPLE:
Conversation has 5 turns, token budget allows only 3 turns:

Collection Phase (Newest-First Priority):
- Evaluates: Turn 5 → Turn 4 → Turn 3 → Turn 2 → Turn 1
- Includes: Turn 5, Turn 4, Turn 3 (newest 3 fit in budget)
- Excludes: Turn 2, Turn 1 (oldest, dropped due to token limits)

Presentation Phase (Chronological Order):
- LLM sees: "--- Turn 3 (Agent) ---", "--- Turn 4 (Model) ---", "--- Turn 5 (Agent) ---"
- Natural conversation flow maintained despite prioritizing recent context

This enables true AI-to-AI collaboration across the entire tool ecosystem with optimal
context preservation and natural conversation understanding.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ValidationError

from utils import transcripts
from utils.env import get_env

logger = logging.getLogger(__name__)


class ConversationMemoryStorageError(RuntimeError):
    """
    Raised when the conversation memory storage backend cannot satisfy a
    request due to a backend fault (connection failure, decode error,
    corrupt payload, etc.) rather than because the requested thread is
    absent.

    Callers should treat this as a transient operational failure and
    surface a "storage error" message to the user, distinct from the
    "thread expired" message used for genuinely missing threads.
    """


# Configuration constants
# Get max conversation turns from environment, default to 20 turns (10 exchanges)
try:
    max_turns_raw = (get_env("MAX_CONVERSATION_TURNS", "50") or "50").strip()
    MAX_CONVERSATION_TURNS = int(max_turns_raw)
    if MAX_CONVERSATION_TURNS <= 0:
        logger.warning(f"Invalid MAX_CONVERSATION_TURNS value ({MAX_CONVERSATION_TURNS}), using default of 50 turns")
        MAX_CONVERSATION_TURNS = 50
except ValueError:
    logger.warning(
        f"Invalid MAX_CONVERSATION_TURNS value ('{get_env('MAX_CONVERSATION_TURNS')}'), using default of 50 turns"
    )
    MAX_CONVERSATION_TURNS = 50

# Get conversation timeout from environment (in hours), default to 3 hours
try:
    timeout_raw = (get_env("CONVERSATION_TIMEOUT_HOURS", "3") or "3").strip()
    CONVERSATION_TIMEOUT_HOURS = int(timeout_raw)
    if CONVERSATION_TIMEOUT_HOURS <= 0:
        logger.warning(
            f"Invalid CONVERSATION_TIMEOUT_HOURS value ({CONVERSATION_TIMEOUT_HOURS}), using default of 3 hours"
        )
        CONVERSATION_TIMEOUT_HOURS = 3
except ValueError:
    logger.warning(
        f"Invalid CONVERSATION_TIMEOUT_HOURS value ('{get_env('CONVERSATION_TIMEOUT_HOURS')}'), using default of 3 hours"
    )
    CONVERSATION_TIMEOUT_HOURS = 3

CONVERSATION_TIMEOUT_SECONDS = CONVERSATION_TIMEOUT_HOURS * 3600

# First line of the history block build_conversation_history emits. Tools check
# for it to tell a prompt that server.py already embedded the history in from
# one they must reconstruct history for (issue #187), so both sides use this.
CONVERSATION_HISTORY_MARKER = "=== CONVERSATION HISTORY (CONTINUATION) ==="

# Heads the current turn's request after the embedded history; a prompt.txt
# request on a continuation is placed right after it.
NEW_USER_INPUT_MARKER = "=== NEW USER INPUT ==="


class ConversationTurn(BaseModel):
    """
    Single turn in a conversation

    Represents one exchange in the AI-to-AI conversation, tracking both
    the content and metadata needed for cross-tool continuation.

    Attributes:
        role: "user" (Agent request) or "assistant" (model response)
        content: The actual message content/response
        timestamp: ISO timestamp when this turn was created
        files: List of file paths referenced in this specific turn
        images: List of image paths referenced in this specific turn
        tool_name: Which tool generated this turn (for cross-tool tracking)
        model_provider: Provider used (e.g., "google", "openai")
        model_name: Specific model used (e.g., "gemini-2.5-flash", "o3-mini")
        model_metadata: Additional model-specific metadata (e.g., thinking mode, token usage)
    """

    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    files: list[str] | None = None  # Files referenced in this turn
    images: list[str] | None = None  # Images referenced in this turn
    tool_name: str | None = None  # Tool used for this turn
    model_provider: str | None = None  # Model provider (google, openai, etc)
    model_name: str | None = None  # Specific model used
    model_metadata: dict[str, Any] | None = None  # Additional model info


class ThreadContext(BaseModel):
    """
    Complete conversation context for a thread

    Contains all information needed to reconstruct a conversation state
    across different tools and request cycles. This is the core data
    structure that enables cross-tool continuation.

    Attributes:
        thread_id: UUID identifying this conversation thread
        parent_thread_id: UUID of parent thread (for conversation chains)
        created_at: ISO timestamp when thread was created
        last_updated_at: ISO timestamp of last modification
        tool_name: Name of the tool that initiated this thread
        turns: List of all conversation turns in chronological order
        initial_context: Original request data that started the conversation
    """

    thread_id: str
    parent_thread_id: str | None = None  # Parent thread for conversation chains
    created_at: str
    last_updated_at: str
    tool_name: str  # Tool that created this thread (preserved for attribution)
    turns: list[ConversationTurn]
    initial_context: dict[str, Any]  # Original request parameters


def get_storage():
    """
    Get in-memory storage backend for conversation persistence.

    Returns:
        InMemoryStorage: Thread-safe in-memory storage backend
    """
    from .storage_backend import get_storage_backend

    return get_storage_backend()


def create_thread(tool_name: str, initial_request: dict[str, Any], parent_thread_id: str | None = None) -> str:
    """
    Create new conversation thread and return thread ID

    Initializes a new conversation thread for AI-to-AI discussions.
    This is called when a tool wants to enable follow-up conversations
    or when Claude explicitly starts a multi-turn interaction.

    Args:
        tool_name: Name of the tool creating this thread (e.g., "analyze", "chat")
        initial_request: Original request parameters (will be filtered for serialization)
        parent_thread_id: Optional parent thread ID for conversation chains

    Returns:
        str: UUID thread identifier that can be used for continuation

    Note:
        - Thread expires after the configured timeout (default: 3 hours)
        - Non-serializable parameters are filtered out automatically
        - Thread can be continued by any tool using the returned UUID
        - Parent thread creates a chain for conversation history traversal
    """
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Filter out non-serializable parameters to avoid JSON encoding issues
    filtered_context = {
        k: v
        for k, v in initial_request.items()
        if k not in ["temperature", "thinking_mode", "model", "continuation_id"]
    }

    context = ThreadContext(
        thread_id=thread_id,
        parent_thread_id=parent_thread_id,  # Link to parent for conversation chains
        created_at=now,
        last_updated_at=now,
        tool_name=tool_name,  # Track which tool initiated this conversation
        turns=[],  # Empty initially, turns added via add_turn()
        initial_context=filtered_context,
    )

    # Store in memory with configurable TTL to prevent indefinite accumulation
    storage = get_storage()
    key = f"thread:{thread_id}"
    storage.setex(key, CONVERSATION_TIMEOUT_SECONDS, context.model_dump_json())
    transcripts.record_thread_created(thread_id, parent_thread_id, tool_name, now)

    logger.debug(f"[THREAD] Created new thread {thread_id} with parent {parent_thread_id}")

    return thread_id


def get_thread(thread_id: str) -> ThreadContext | None:
    """
    Retrieve thread context from in-memory storage

    Fetches complete conversation context for cross-tool continuation.
    This is the core function that enables tools to access conversation
    history from previous interactions.

    Args:
        thread_id: UUID of the conversation thread

    Returns:
        ThreadContext: Complete conversation context if found
        None: If thread doesn't exist, expired, or invalid UUID

    Raises:
        ConversationMemoryStorageError: When the storage backend itself
            fails (connection error, decode failure, corrupt payload).
            This is distinct from the absent-thread case, which returns
            ``None`` silently. Callers should surface a "storage error"
            message to users rather than "thread expired".

    Security:
        - Validates UUID format to prevent injection attacks
        - Backend faults are logged at WARNING with ``exc_info=True`` so
          they appear in ``mcp_server.log`` for operators, but the
          original exception details are not propagated to end users
          beyond the typed ``ConversationMemoryStorageError`` wrapper.
    """
    if not thread_id or not _is_valid_uuid(thread_id):
        return None

    key = f"thread:{thread_id}"

    try:
        storage = get_storage()
        data = storage.get(key)
    except Exception as exc:
        # Backend fault (storage outage, lock error, etc.) — log loudly
        # so operators can see the real cause in mcp_server.log, and
        # signal a distinct failure mode to callers rather than
        # collapsing it into "thread not found".
        logger.warning(
            f"[THREAD] Storage backend failure while retrieving {thread_id}: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        raise ConversationMemoryStorageError(
            f"Conversation storage backend failure while retrieving thread {thread_id}"
        ) from exc

    # Absent-thread case: storage returned no value. This is the expected
    # path for an expired or never-created thread and stays silent.
    if not data:
        return None

    try:
        return ThreadContext.model_validate_json(data)
    except ValidationError as exc:
        # Stored payload is unparseable — corrupt entry, schema drift,
        # or someone wrote garbage into the backend. Treat as a backend
        # fault, not a missing thread.
        logger.warning(
            f"[THREAD] Corrupt thread payload for {thread_id}: {exc}",
            exc_info=True,
        )
        raise ConversationMemoryStorageError(
            f"Conversation storage payload for thread {thread_id} could not be decoded"
        ) from exc


def add_turn(
    thread_id: str,
    role: str,
    content: str,
    files: list[str] | None = None,
    images: list[str] | None = None,
    tool_name: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> bool:
    """
    Add turn to existing thread with atomic file ordering.

    Appends a new conversation turn to an existing thread. This is the core
    function for building conversation history and enabling cross-tool
    continuation. Each turn preserves the tool and model that generated it.

    Args:
        thread_id: UUID of the conversation thread
        role: "user" (Agent request) or "assistant" (model response)
        content: The actual message/response content
        files: Optional list of files referenced in this turn
        images: Optional list of images referenced in this turn
        tool_name: Name of the tool adding this turn (for attribution)
        model_provider: Provider used (e.g., "google", "openai")
        model_name: Specific model used (e.g., "gemini-2.5-flash", "o3-mini")
        model_metadata: Additional model info (e.g., thinking mode, token usage)

    Returns:
        bool: True if turn was successfully added. ``False`` is returned
        only for *expected* non-fault outcomes: the thread is absent
        (expired / never created) or the thread has reached the
        ``MAX_CONVERSATION_TURNS`` cap.

    Raises:
        ConversationMemoryStorageError: When the storage backend itself
            fails — either while loading the thread (propagated from
            ``get_thread``) or while writing the updated thread back.
            These are hard storage faults that must not be silently
            swallowed, since the caller's notion of "turn saved" would
            be wrong. The fault is logged at WARNING with
            ``exc_info=True`` so it appears in ``mcp_server.log`` for
            operators.

    Note:
        - Refreshes thread TTL to configured timeout on successful update
        - Turn limits prevent runaway conversations
        - File references are preserved for cross-tool access with atomic ordering
        - Image references are preserved for cross-tool visual context
        - Model information enables cross-provider conversations
    """
    logger.debug(f"[FLOW] Adding {role} turn to {thread_id} ({tool_name})")

    # Let ConversationMemoryStorageError from get_thread propagate — it
    # represents a hard storage fault, not a "thread doesn't exist"
    # outcome. get_thread has already emitted a WARNING with exc_info.
    context = get_thread(thread_id)
    if not context:
        logger.debug(f"[FLOW] Thread {thread_id} not found for turn addition")
        return False

    # Check turn limit to prevent runaway conversations
    if len(context.turns) >= MAX_CONVERSATION_TURNS:
        logger.debug(f"[FLOW] Thread {thread_id} at max turns ({MAX_CONVERSATION_TURNS})")
        return False

    # Create new turn with complete metadata
    turn = ConversationTurn(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        files=files,  # Preserved for cross-tool file context
        images=images,  # Preserved for cross-tool visual context
        tool_name=tool_name,  # Track which tool generated this turn
        model_provider=model_provider,  # Track model provider
        model_name=model_name,  # Track specific model
        model_metadata=model_metadata,  # Additional model info
    )

    context.turns.append(turn)
    context.last_updated_at = datetime.now(timezone.utc).isoformat()

    # Save back to storage and refresh TTL.
    try:
        storage = get_storage()
        key = f"thread:{thread_id}"
        storage.setex(key, CONVERSATION_TIMEOUT_SECONDS, context.model_dump_json())  # Refresh TTL to configured timeout
    except Exception as exc:
        # Storage write failure is a hard backend fault, not a
        # validation outcome. Log loudly so operators see the real
        # cause in mcp_server.log and raise the typed error so
        # callers can distinguish it from "thread expired" / "at
        # turn limit". Returning False here would silently lose the
        # turn from the agent's perspective.
        logger.warning(
            f"[FLOW] Storage backend failure while saving turn to thread {thread_id}: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
        raise ConversationMemoryStorageError(
            f"Conversation storage backend failure while saving turn to thread {thread_id}"
        ) from exc

    # Transcript writes never raise; a failure is logged and the turn stands.
    transcripts.record_turn(thread_id, turn)
    return True


def get_thread_chain(thread_id: str, max_depth: int = 20) -> list[ThreadContext]:
    """
    Traverse the parent chain to get all threads in conversation sequence.

    Retrieves the complete conversation chain by following parent_thread_id
    links. Returns threads in chronological order (oldest first).

    Args:
        thread_id: Starting thread ID
        max_depth: Maximum chain depth to prevent infinite loops

    Returns:
        list[ThreadContext]: All threads in chain, oldest first
    """
    chain = []
    current_id = thread_id
    seen_ids = set()

    # Build chain from current to oldest
    while current_id and len(chain) < max_depth:
        # Prevent circular references
        if current_id in seen_ids:
            logger.warning(f"[THREAD] Circular reference detected in thread chain at {current_id}")
            break

        seen_ids.add(current_id)

        try:
            context = get_thread(current_id)
        except ConversationMemoryStorageError:
            # Storage failure during chain traversal — get_thread has
            # already logged the underlying fault. Stop walking the
            # chain rather than reporting a misleading "not found".
            logger.warning(f"[THREAD] Storage failure interrupted chain traversal at {current_id}")
            break
        if not context:
            logger.debug(f"[THREAD] Thread {current_id} not found in chain traversal")
            break

        chain.append(context)
        current_id = context.parent_thread_id

    # Reverse to get chronological order (oldest first)
    chain.reverse()

    logger.debug(f"[THREAD] Retrieved chain of {len(chain)} threads for {thread_id}")
    return chain


def get_conversation_file_list(context: ThreadContext) -> list[str]:
    """
    Extract all unique files from conversation turns with newest-first prioritization.

    This function implements the core file prioritization logic used throughout the
    conversation memory system. It walks backwards through conversation turns
    (from newest to oldest) and collects unique file references, ensuring that
    when the same file appears in multiple turns, the reference from the NEWEST
    turn takes precedence.

    PRIORITIZATION ALGORITHM:
    1. Iterate through turns in REVERSE order (index len-1 down to 0)
    2. For each turn, process files in the order they appear in turn.files
    3. Add file to result list only if not already seen (newest reference wins)
    4. Skip duplicate files that were already added from newer turns

    This ensures that:
    - Files from newer conversation turns appear first in the result
    - When the same file is referenced multiple times, only the newest reference is kept
    - The order reflects the most recent conversation context

    Example:
        Turn 1: files = ["main.py", "utils.py"]
        Turn 2: files = ["test.py"]
        Turn 3: files = ["main.py", "config.py"]  # main.py appears again

        Result: ["main.py", "config.py", "test.py", "utils.py"]
        (main.py from Turn 3 takes precedence over Turn 1)

    Args:
        context: ThreadContext containing all conversation turns to process

    Returns:
        list[str]: Unique file paths ordered by newest reference first.
                   Empty list if no turns exist or no files are referenced.

    Performance:
        - Time Complexity: O(n*m) where n=turns, m=avg files per turn
        - Space Complexity: O(f) where f=total unique files
        - Uses set for O(1) duplicate detection
    """
    if not context.turns:
        logger.debug("[FILES] No turns found, returning empty file list")
        return []

    # Collect files by walking backwards (newest to oldest turns)
    seen_files = set()
    file_list = []

    logger.debug(f"[FILES] Collecting files from {len(context.turns)} turns (newest first)")

    # Process turns in reverse order (newest first) - this is the CORE of newest-first prioritization
    # By iterating from len-1 down to 0, we encounter newer turns before older turns
    # When we find a duplicate file, we skip it because the newer version is already in our list
    for i in range(len(context.turns) - 1, -1, -1):  # REVERSE: newest turn first
        turn = context.turns[i]
        if turn.files:
            logger.debug(f"[FILES] Turn {i + 1} has {len(turn.files)} files: {turn.files}")
            for file_path in turn.files:
                if file_path not in seen_files:
                    # First time seeing this file - add it (this is the NEWEST reference)
                    seen_files.add(file_path)
                    file_list.append(file_path)
                    logger.debug(f"[FILES] Added new file: {file_path} (from turn {i + 1})")
                else:
                    # File already seen from a NEWER turn - skip this older reference
                    logger.debug(f"[FILES] Skipping duplicate file: {file_path} (newer version already included)")

    logger.debug(f"[FILES] Final file list ({len(file_list)}): {file_list}")
    return file_list


def get_conversation_image_list(context: ThreadContext) -> list[str]:
    """
    Extract all unique images from conversation turns with newest-first prioritization.

    This function implements the identical prioritization logic as get_conversation_file_list()
    to ensure consistency in how images are handled across conversation turns. It walks
    backwards through conversation turns (from newest to oldest) and collects unique image
    references, ensuring that when the same image appears in multiple turns, the reference
    from the NEWEST turn takes precedence.

    PRIORITIZATION ALGORITHM:
    1. Iterate through turns in REVERSE order (index len-1 down to 0)
    2. For each turn, process images in the order they appear in turn.images
    3. Add image to result list only if not already seen (newest reference wins)
    4. Skip duplicate images that were already added from newer turns

    This ensures that:
    - Images from newer conversation turns appear first in the result
    - When the same image is referenced multiple times, only the newest reference is kept
    - The order reflects the most recent conversation context

    Example:
        Turn 1: images = ["diagram.png", "flow.jpg"]
        Turn 2: images = ["error.png"]
        Turn 3: images = ["diagram.png", "updated.png"]  # diagram.png appears again

        Result: ["diagram.png", "updated.png", "error.png", "flow.jpg"]
        (diagram.png from Turn 3 takes precedence over Turn 1)

    Args:
        context: ThreadContext containing all conversation turns to process

    Returns:
        list[str]: Unique image paths ordered by newest reference first.
                   Empty list if no turns exist or no images are referenced.

    Performance:
        - Time Complexity: O(n*m) where n=turns, m=avg images per turn
        - Space Complexity: O(i) where i=total unique images
        - Uses set for O(1) duplicate detection
    """
    if not context.turns:
        logger.debug("[IMAGES] No turns found, returning empty image list")
        return []

    # Collect images by walking backwards (newest to oldest turns)
    seen_images = set()
    image_list = []

    logger.debug(f"[IMAGES] Collecting images from {len(context.turns)} turns (newest first)")

    # Process turns in reverse order (newest first) - this is the CORE of newest-first prioritization
    # By iterating from len-1 down to 0, we encounter newer turns before older turns
    # When we find a duplicate image, we skip it because the newer version is already in our list
    for i in range(len(context.turns) - 1, -1, -1):  # REVERSE: newest turn first
        turn = context.turns[i]
        if turn.images:
            logger.debug(f"[IMAGES] Turn {i + 1} has {len(turn.images)} images: {turn.images}")
            for image_path in turn.images:
                if image_path not in seen_images:
                    # First time seeing this image - add it (this is the NEWEST reference)
                    seen_images.add(image_path)
                    image_list.append(image_path)
                    logger.debug(f"[IMAGES] Added new image: {image_path} (from turn {i + 1})")
                else:
                    # Image already seen from a NEWER turn - skip this older reference
                    logger.debug(f"[IMAGES] Skipping duplicate image: {image_path} (newer version already included)")

    logger.debug(f"[IMAGES] Final image list ({len(image_list)}): {image_list}")
    return image_list


def _plan_file_inclusion_by_size(all_files: list[str], max_file_tokens: int) -> tuple[list[str], list[str], int]:
    """
    Plan which files to include based on size constraints.

    This is ONLY used for conversation history building, not MCP boundary checks.

    Args:
        all_files: List of files to consider for inclusion
        max_file_tokens: Maximum tokens available for file content

    Returns:
        Tuple of (files_to_include, files_to_skip, estimated_total_tokens)
    """
    if not all_files:
        return [], [], 0

    files_to_include = []
    files_to_skip = []
    total_tokens = 0

    logger.debug(f"[FILES] Planning inclusion for {len(all_files)} files with budget {max_file_tokens:,} tokens")

    for file_path in all_files:
        try:
            from utils.file_utils import estimate_file_tokens

            if os.path.exists(file_path) and os.path.isfile(file_path):
                # Use centralized token estimation for consistency
                estimated_tokens = estimate_file_tokens(file_path)

                if total_tokens + estimated_tokens <= max_file_tokens:
                    files_to_include.append(file_path)
                    total_tokens += estimated_tokens
                    logger.debug(
                        f"[FILES] Including {file_path} - {estimated_tokens:,} tokens (total: {total_tokens:,})"
                    )
                else:
                    files_to_skip.append(file_path)
                    logger.debug(
                        f"[FILES] Skipping {file_path} - would exceed budget (needs {estimated_tokens:,} tokens)"
                    )
            else:
                files_to_skip.append(file_path)
                # More descriptive message for missing files
                if not os.path.exists(file_path):
                    logger.debug(
                        f"[FILES] Skipping {file_path} - file no longer exists (may have been moved/deleted since conversation)"
                    )
                else:
                    logger.debug(f"[FILES] Skipping {file_path} - file not accessible (not a regular file)")

        except Exception as e:
            files_to_skip.append(file_path)
            logger.debug(f"[FILES] Skipping {file_path} - error during processing: {type(e).__name__}: {e}")

    logger.debug(
        f"[FILES] Inclusion plan: {len(files_to_include)} include, {len(files_to_skip)} skip, {total_tokens:,} tokens"
    )
    return files_to_include, files_to_skip, total_tokens


def _history_turns_and_files(context: ThreadContext) -> tuple[list[ConversationTurn], list[str]]:
    """Return the turns the history covers and the files they list, newest file first.

    A thread with a parent covers its whole chain, so files from newer turns in
    any thread of it take precedence over older ones.
    """
    if not context.parent_thread_id:
        return context.turns, get_conversation_file_list(context)

    chain = get_thread_chain(context.thread_id)
    all_turns = [turn for thread in chain for turn in thread.turns]
    merged = ThreadContext(
        thread_id="merged_chain",
        created_at=context.created_at,
        last_updated_at=context.last_updated_at,
        tool_name=context.tool_name,
        turns=all_turns,  # All turns from entire chain in chronological order
        initial_context=context.initial_context,
    )
    logger.debug(f"[THREAD] Built history from {len(chain)} threads with {len(all_turns)} total turns")
    return all_turns, get_conversation_file_list(merged)


def get_history_embedded_files(context: ThreadContext, model_context) -> list[str]:
    """Return the files ``build_conversation_history`` embeds for this thread and model.

    A listed file the history's file budget cannot hold is left out of the
    history, so a tool must embed it itself rather than treat it as embedded.
    """
    _, all_files = _history_turns_and_files(context)
    files_to_include, _, _ = _plan_file_inclusion_by_size(
        all_files, model_context.calculate_token_allocation().file_tokens
    )
    return files_to_include


#: Share of the turn budget left filled after a compaction drops old turns (issue #195).
COMPACTION_KEEP_RATIO = 0.6
#: Ends the note that replaces the turns a compaction dropped.
OMITTED_TURNS_NOTE = "omitted to fit the history budget"
#: Heads the files attached with the request being answered, after the last turn block.
NEW_INPUT_FILES_HEADER = "Files attached with the new input:"


def _compaction_start(turn_tokens: list[int], budget: int) -> int:
    """Index of the first turn to render, found by replaying compaction over the turns in order.

    Whenever the turns kept so far outgrow ``budget``, the oldest are dropped until
    they fill at most ``COMPACTION_KEEP_RATIO`` of it. Replaying the whole sequence
    makes the result a function of the turns alone: request k+1 reaches the same
    start as request k until the next compaction, so the rendered history is
    rewritten once per compaction instead of on every call after the first
    overflow, as a newest-first fill did (issue #195).
    """
    start = 0
    total = 0
    for index, tokens in enumerate(turn_tokens):
        total += tokens
        if total > budget:
            while start <= index and total > budget * COMPACTION_KEEP_RATIO:
                total -= turn_tokens[start]
                start += 1
    return start


def _turn_header(turn: ConversationTurn, turn_num: int) -> str:
    """The ``--- Turn N (...) ---`` line, with tool and model attribution for cross-tool threads."""
    role_label = "Agent" if turn.role == "user" else (turn.model_name or "Assistant")
    header = f"\n--- Turn {turn_num} ({role_label}"
    if turn.tool_name:
        header += f" using {turn.tool_name}"
    if turn.model_provider:
        provider_descriptor = turn.model_provider
        if turn.model_name and turn.model_name != role_label:
            provider_descriptor += f"/{turn.model_name}"
        header += f" via {provider_descriptor}"
    elif turn.model_name and turn.model_name != role_label:
        header += f" via {turn.model_name}"
    return header + ") ---"


def _read_history_files(paths: list[str]) -> dict[str, str]:
    """Formatted content of each readable file, keyed by path; unreadable files are left out."""
    from utils.file_utils import read_file_content

    contents: dict[str, str] = {}
    for file_path in paths:
        try:
            formatted, _ = read_file_content(file_path)
        except Exception as e:
            logger.info(f"File not embedded in conversation history: {file_path} - {type(e).__name__}: {e}")
            continue
        if formatted:
            contents[file_path] = formatted
    return contents


def _file_blocks(paths: list[str], contents: dict[str, str], shown: dict[str, str], where: str) -> list[str]:
    """Render the files a turn lists, recording in ``shown`` where each first appeared.

    A file appears in full at its first reference, with its current content, and
    as a one-line pointer after that. A file the budget left out, or that cannot
    be read, is a one-line marker at its first reference.
    """
    blocks = []
    for file_path in paths:
        if file_path in shown:
            blocks.append(f"[File {file_path}: see {shown[file_path]}]")
        elif file_path in contents:
            blocks.append(contents[file_path])
            shown[file_path] = where
        else:
            blocks.append(f"[File {file_path} omitted: over the history file budget, missing, or unreadable]")
            shown[file_path] = f"the note in {where}"
    return blocks


def build_conversation_history(
    context: ThreadContext, model_context=None, *, latest_turn_is_input: bool = False
) -> tuple[str, int]:
    """
    Build the conversation history for a continuation, with the files its turns attached.

    The history is laid out so that consecutive requests of a thread share as long
    a prefix as possible, which is what provider prompt caches reuse
    (docs/prompt-caching.md, Stages A and B):

    - The header names the thread and tool and carries no turn counter.
    - Turns appear in order. Each file appears inside the turn that first listed it,
      read from disk on every call, and as a one-line pointer in later turns. Adding
      a file therefore appends to the history instead of changing a files section
      ahead of every turn.
    - Files are chosen for the file budget newest reference first, so when files
      must be left out the oldest go; a left-out file is a one-line marker.
    - When the turns outgrow the history budget, old turns are dropped in one step
      down to ``COMPACTION_KEEP_RATIO`` of it, and a note says which. The files of
      dropped turns stay, in a block after that note.

    A thread with a parent covers the whole chain.

    Args:
        context: The thread to render.
        model_context: ModelContext for the token allocation (DEFAULT_MODEL when omitted).
        latest_turn_is_input: The thread's last turn is the request being answered,
            recorded before the history is built so its files reach the model on
            this turn. Its files are rendered after the last turn block, but its text
            is left to the caller's NEW USER INPUT slot, so the request is not sent
            twice (issue #194).

    Returns:
        ``(history, estimated_tokens)``, or ``("", 0)`` for a thread with no turns.
    """
    all_turns, all_files = _history_turns_and_files(context)

    if not all_turns:
        return "", 0

    current_turn = None
    if latest_turn_is_input and all_turns[-1].role == "user":
        all_turns, current_turn = all_turns[:-1], all_turns[-1]

    if model_context is None:
        from config import DEFAULT_MODEL, IS_AUTO_MODE
        from utils.model_context import ModelContext

        model_name = DEFAULT_MODEL
        if IS_AUTO_MODE and model_name.lower() == "auto":
            from providers.registry import ModelProviderRegistry

            model_name = ModelProviderRegistry.get_preferred_fallback_model()

        model_context = ModelContext(model_name)

    token_allocation = model_context.calculate_token_allocation()
    files_to_include, files_to_skip, _ = _plan_file_inclusion_by_size(all_files, token_allocation.file_tokens)
    if files_to_skip:
        logger.info(f"[FILES] Excluding {len(files_to_skip)} files from conversation history: {files_to_skip}")
    contents = _read_history_files(files_to_include)

    header = [
        CONVERSATION_HISTORY_MARKER,
        f"Thread: {context.thread_id}",
        f"Tool: {context.tool_name}",  # Original tool that started the conversation
        # No turn counter here: it would be the first byte to change between
        # consecutive requests, and providers cache only an unchanged prefix (#194).
        "You are continuing this conversation thread from where it left off.",
        "",
        "Previous conversation turns:",
    ]

    # Turn text competes with the files for the history budget; the files are a
    # fixed cost, so compaction works on what they leave.
    turn_texts = [
        "\n".join([_turn_header(turn, i + 1), *_get_tool_formatted_content(turn)]) for i, turn in enumerate(all_turns)
    ]
    fixed_tokens = sum(model_context.estimate_tokens(part) for part in [*header, *contents.values()])
    turn_budget = token_allocation.history_tokens - fixed_tokens
    start = _compaction_start([model_context.estimate_tokens(text) for text in turn_texts], turn_budget)

    shown: dict[str, str] = {}
    body: list[str] = []
    if start:
        logger.info(f"[HISTORY] Turns 1-{start} of {len(all_turns)} omitted by compaction")
        body.append(f"\n[Note: Turns 1-{start} {OMITTED_TURNS_NOTE}]")
        omitted_files = list(dict.fromkeys(f for turn in all_turns[:start] for f in turn.files or []))
        if omitted_files:
            body.append("Files from the omitted turns:")
            body.extend(_file_blocks(omitted_files, contents, shown, "the files from the omitted turns"))
    for index in range(start, len(all_turns)):
        body.append(turn_texts[index])
        body.extend(_file_blocks(all_turns[index].files or [], contents, shown, f"Turn {index + 1}"))

    trailer = [
        "",
        "=== END CONVERSATION HISTORY ===",
        "",
        "IMPORTANT: You are continuing an existing conversation thread. Build upon the previous exchanges shown above,",
        "reference earlier points, and maintain consistency with what has been discussed.",
        "",
        "DO NOT repeat or summarize previous analysis, findings, or instructions that are already covered in the",
        "conversation history. Instead, provide only new insights, additional analysis, or direct answers to",
        "the follow-up question / concerns / insights. Assume the user has read the prior conversation.",
        "",
        f"This is turn {len(all_turns) + 1} of the conversation - use the conversation history above to provide a coherent continuation.",
    ]

    # The new input's files go after the trailer, beside the new input, not after
    # the last turn block: on the next request that position holds this turn's
    # block, and the history up to it must not change.
    if current_turn is not None and current_turn.files:
        trailer.append(f"\n{NEW_INPUT_FILES_HEADER}")
        trailer.extend(_file_blocks(current_turn.files, contents, shown, "the new input"))

    complete_history = "\n".join([*header, *body, *trailer])
    from utils.token_utils import estimate_tokens

    total_conversation_tokens = estimate_tokens(complete_history)
    logger.debug(
        f"[FLOW] Built conversation history: {len(all_turns) - start} of {len(all_turns)} turns, "
        f"{len(contents)} files, {total_conversation_tokens:,} tokens"
    )
    return complete_history, total_conversation_tokens


def _get_tool_formatted_content(turn: ConversationTurn) -> list[str]:
    """
    Get tool-specific formatting for a conversation turn.

    This function attempts to use the tool's custom formatting method if available,
    falling back to default formatting if the tool cannot be found or doesn't
    provide custom formatting.

    Args:
        turn: The conversation turn to format

    Returns:
        list[str]: Formatted content lines for this turn
    """
    if turn.tool_name:
        try:
            # Dynamically import to avoid circular dependencies
            from server import TOOLS

            tool = TOOLS.get(turn.tool_name)
            if tool:
                # Use inheritance pattern - try to call the method directly
                # If it doesn't exist or raises AttributeError, fall back to default
                try:
                    return tool.format_conversation_turn(turn)
                except AttributeError:
                    # Tool doesn't implement format_conversation_turn - use default
                    pass
        except Exception as e:
            # Log but don't fail - fall back to default formatting
            logger.debug(f"[HISTORY] Could not get tool-specific formatting for {turn.tool_name}: {e}")

    # Default formatting
    return _default_turn_formatting(turn)


def _default_turn_formatting(turn: ConversationTurn) -> list[str]:
    """
    Default formatting for conversation turns.

    This provides the standard formatting when no tool-specific
    formatting is available.

    Args:
        turn: The conversation turn to format

    Returns:
        list[str]: Default formatted content lines
    """
    parts = []

    # Add files context if present
    if turn.files:
        parts.append(f"Files used in this turn: {', '.join(turn.files)}")
        parts.append("")  # Empty line for readability

    # Add the actual content
    parts.append(turn.content)

    return parts


def _is_valid_uuid(val: str) -> bool:
    """
    Validate UUID format for security

    Ensures thread IDs are valid UUIDs to prevent injection attacks
    and malformed requests.

    Args:
        val: String to validate as UUID

    Returns:
        bool: True if valid UUID format, False otherwise
    """
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False
