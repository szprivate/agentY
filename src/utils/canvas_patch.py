"""canvas_patch – thread-safe buffer for node edits the agent pushes back to the
live ComfyUI canvas.

When the orchestrator alters a selected node's parameters (via the
``set_canvas_node_params`` tool), it pushes a small patch dict here. The pipeline
drains the buffer in its stream loop and yields ``{"canvas_patch": {...}}``
events; the chat host forwards them over the open SSE stream and
``web/agent_chat.js`` applies them to the live graph — no browser refresh, no
re-queue (the panel runs in-page and holds an ``app.graph`` reference).

Mirrors ``src.utils.tool_activity``: same push / drain / clear contract.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

_lock: threading.Lock = threading.Lock()
# Bounded so a run whose consumer stops draining can't grow without limit.
_events: deque[dict[str, Any]] = deque(maxlen=200)


def push(event: dict[str, Any]) -> None:
    """Append a canvas-patch event (thread-safe)."""
    with _lock:
        _events.append(event)


def drain() -> list[dict[str, Any]]:
    """Atomically read and clear all buffered patches (empty list if none)."""
    with _lock:
        if not _events:
            return []
        out = list(_events)
        _events.clear()
        return out


def clear() -> None:
    """Discard any buffered patches (call at the start of a turn)."""
    with _lock:
        _events.clear()
