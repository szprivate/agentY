"""
tool_activity – thread-safe buffer for agent tool-call activity.

The orchestrator's ``ToolActivityHookProvider`` (src/agent.py) pushes a small
dict for every tool call (before) and result (after) here; the pipeline drains
them in its stream loop and yields ``{"tool_activity": {...}}`` events so the
ComfyUI chat UI can render the agent's tool use inline in the conversation.

Mirrors ``agenty_core.utils.progress_signal`` but carries structured dicts.

One buffer per turn (:mod:`agenty_core.utils.turn_scope`): with several
conversations running at once, each turn's stream forwards its own events only.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from agenty_core.utils import turn_scope

# Bounded so a run whose consumer stops draining can't grow without limit.
_MAX = 500


def _events(scope) -> deque:
    return scope.slot("tool_activity", lambda: deque(maxlen=_MAX))


def push(event: dict[str, Any]) -> None:
    """Append a tool-activity event to the current turn's buffer (thread-safe)."""
    scope = turn_scope.current()
    with scope.lock:
        _events(scope).append(event)


def drain(scope=None) -> list[dict[str, Any]]:
    """Atomically read and clear the buffered events of *scope* (default: the
    current turn's; empty list if none)."""
    scope = scope or turn_scope.current()
    with scope.lock:
        buf = _events(scope)
        if not buf:
            return []
        out = list(buf)
        buf.clear()
        return out


def clear(scope=None) -> None:
    """Discard the buffered events of *scope* (call at the start of a turn)."""
    scope = scope or turn_scope.current()
    with scope.lock:
        _events(scope).clear()
