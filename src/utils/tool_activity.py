"""
tool_activity – thread-safe buffer for agent tool-call activity.

The orchestrator's ``ToolActivityHookProvider`` (src/agent.py) pushes a small
dict for every tool call (before) and result (after) here; the pipeline drains
them in its stream loop and yields ``{"tool_activity": {...}}`` events so the
ComfyUI chat UI can render the agent's tool use inline in the conversation.

Mirrors ``agenty_core.utils.progress_signal`` but carries structured dicts.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

_lock: threading.Lock = threading.Lock()
# Bounded so a run whose consumer stops draining can't grow without limit.
_events: deque[dict[str, Any]] = deque(maxlen=500)


def push(event: dict[str, Any]) -> None:
    """Append a tool-activity event (thread-safe)."""
    with _lock:
        _events.append(event)


def drain() -> list[dict[str, Any]]:
    """Atomically read and clear all buffered events (empty list if none)."""
    with _lock:
        if not _events:
            return []
        out = list(_events)
        _events.clear()
        return out


def clear() -> None:
    """Discard any buffered events (call at the start of a turn)."""
    with _lock:
        _events.clear()
