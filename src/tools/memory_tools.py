"""
agentY – Strands @tool wrappers for the local FAISS memory layer.

These two tools are injected into both the Query Templates and Assemble Workflow agents so
they can read from and write to long-term memory during a run.

Session binding
---------------
The pipeline calls ``set_session_id(session_id)`` before each run so both
tools operate on the correct per-user/per-session memory namespace without
requiring it as an argument in every LLM call.
"""

from __future__ import annotations

from strands import tool

from src.utils.memory import MEMORY_NAMESPACE, format_memories, memory_add, memory_search

# ---------------------------------------------------------------------------
# Module-level session binding
# The Pipeline sets this once via set_session_id() so individual tool calls
# don't need to carry the session_id as an LLM-visible parameter.
#
# NOTE: both memory tools now read from and write to the single shared
# ``MEMORY_NAMESPACE`` (so a lesson learned in one session is recalled in every
# other), not this per-session id. The binding is retained for any caller that
# still relies on it, but memory_read / memory_write ignore it by design.
# ---------------------------------------------------------------------------

_SESSION_ID: str = "default"


def set_session_id(session_id: str) -> None:
    """Update the active session ID for both memory tools.

    Must be called by the Pipeline after construction and whenever the
    session_id changes (e.g. a new Chainlit conversation is started).
    """
    global _SESSION_ID
    _SESSION_ID = session_id


# ---------------------------------------------------------------------------
# Strands tools
# ---------------------------------------------------------------------------

@tool
def memory_read(query: str) -> str:
    """Search long-term memory for facts relevant to the current task.

    Call this at the **start of a new task** to recall:
    - User style or quality preferences (e.g. preferred resolution, aspect ratio)
    - Workflow or model choices that worked well in past sessions
    - Subject matter details the user has shared before (character names, colour
      palettes, recurring prompts)
    - Any notes the user asked to remember

    Args:
        query: Natural-language description of what to look for in memory,
               e.g. "user portrait resolution preference" or "LoRA for anime style".

    Returns:
        A formatted list of relevant memories, or a message indicating no
        matches were found.
    """
    results = memory_search(query, session_id=MEMORY_NAMESPACE, limit=5)
    text = format_memories(results)
    if not text:
        return "(no relevant memories found)"
    return text


@tool
def memory_write(content: str) -> str:
    """Persist a fact, preference, or workflow note to long-term memory.

    Call this **after resolving something worth remembering**, such as:
    - A user preference discovered during this session ("User wants 768×1344
      for all portrait outputs")
    - A model or LoRA combination that produced good results
    - A workflow template the user liked or disliked
    - Any explicit note the user asked to save ("remember that my character
      wears a red cloak")

    Write one clear, self-contained sentence per call.  Do NOT call this
    for ephemeral working notes — only for facts worth carrying across sessions.

    Args:
        content: The memory to store, written as a complete, standalone sentence.
                 Example: "User prefers cinematic wide shots at 1920×816 for
                 all exterior scene workflows."

    Returns:
        Confirmation of what was actually stored (or a clear notice if it was not).
    """
    text = (content or "").strip()
    if not text:
        return "Nothing to store — the content was empty."
    # Store verbatim (memory_add defaults to infer=False) so the exact sentence
    # persists, and inspect the result rather than assuming success: mem0 returns
    # the rows it created, so we can tell the model the truth about what landed.
    result = memory_add(text, session_id=MEMORY_NAMESPACE, metadata={"source": "user"})
    stored = isinstance(result, dict) and any(
        r.get("event") == "ADD" for r in result.get("results", [])
    )
    if stored:
        return f"Stored in long-term memory: {text}"
    return (
        "Could not confirm the memory was stored — the memory layer returned no new "
        f"entry (it may be disabled). NOT saved: {text}"
    )
