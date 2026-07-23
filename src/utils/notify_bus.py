"""agentY notification bus — structured, between-turn events for the sidebar.

Sibling of :mod:`status_bus`, but for **structured** notifications rather than
plain status text. Its first use is Magnific's background auto-drop: when an
async creation finishes (minutes after the turn that queued it ended), the
watcher (:mod:`src.utils.magnific_watch`) emits a ``media`` event here carrying
the downloaded file so the panel can drop a loader node onto the ComfyUI canvas
and raise a pop-up — all while the chat is idle.

Delivery mirrors ``status_bus``: a bounded ring buffer + monotonic ``seq`` the
panel drains via ``GET /agentY/notifications?since=<seq>`` (the idle path, since
there is no live SSE stream between turns), plus live fan-out to any registered
per-turn queue so a completion that lands mid-turn streams immediately. The panel
dedupes the two paths by ``seq``.

Event shape (all events)::

    {"seq": int, "ts": float, "kind": "media"|"toast"|"error",
     "toast": {"title", "body", "url", "level"},
     "output": {"kind", "path", "filename", "name", "node_candidates"}}  # kind=="media"

Kept stdlib-only so low-level modules can import it without an import cycle.
"""
from __future__ import annotations

import threading
import time
from collections import deque

_LOCK = threading.Lock()
_MAX = 100
_buffer: "deque[dict]" = deque(maxlen=_MAX)
_seq = 0
_listeners: set = set()  # live queue.Queue objects (one per in-flight turn)


def emit(event: dict) -> dict:
    """Record a structured notification and fan it out live to every listener.

    *event* is a dict (see the module docstring); ``seq`` and ``ts`` are assigned
    here. Returns the stored event. Live listeners receive it wrapped as an SSE
    frame ``{"type": "notify", ...event}``.
    """
    global _seq
    if not isinstance(event, dict):
        event = {"kind": "toast", "toast": {"title": "agentY", "body": str(event)}}
    with _LOCK:
        _seq += 1
        item = dict(event)
        item["seq"] = _seq
        item["ts"] = time.time()
        _buffer.append(item)
        listeners = list(_listeners)
    frame = {"type": "notify", **item}
    for q in listeners:
        try:
            q.put(frame)
        except Exception:  # noqa: BLE001 — a wedged listener must never break others
            pass
    return item


def snapshot(since: int = 0) -> dict:
    """Buffered events with ``seq > since`` (in order) plus the latest seq.

    The panel passes the highest seq it has already handled; this returns only
    the events it is missing, so the idle poll never re-drops a node already
    delivered live during a turn.
    """
    with _LOCK:
        items = [dict(i) for i in _buffer if i["seq"] > since]
        latest = _seq
    return {"seq": latest, "events": items}


def register_live(q) -> None:
    """Register a queue to receive live ``notify`` events (one per turn)."""
    with _LOCK:
        _listeners.add(q)


def unregister_live(q) -> None:
    with _LOCK:
        _listeners.discard(q)
