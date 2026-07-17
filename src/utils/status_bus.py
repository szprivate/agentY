"""agentY status bus — surface CLI-side pipeline notices in the sidebar too.

A handful of pipeline / startup notices (e.g. the FAISS memory layer
initialising) are printed to the console that runs ``run_agent.ps1`` but never
reached the ComfyUI sidebar. This tiny bus fixes that: notable notices call
:func:`notify`, which both prints to the console (unchanged CLI behaviour) *and*
records the line in a bounded ring buffer + fans it out to any live SSE listener.

The sidebar drains the ring buffer on connect (so startup lines that predate the
panel still show) and receives live ``status_line`` events during a turn. Each
line carries a monotonic ``seq`` so the panel can dedupe the live path against
the on-connect / on-done buffer fetch (``GET /agentY/status?since=<seq>``).

Kept dependency-free (stdlib only) so any module — including low-level ones like
``memory`` — can import it without risking an import cycle.
"""
from __future__ import annotations

import threading
import time
from collections import deque

_LOCK = threading.Lock()
_MAX = 200
_buffer: "deque[dict]" = deque(maxlen=_MAX)
_seq = 0
_listeners: set = set()  # live queue.Queue objects (one per in-flight turn)


def emit(text: str, *, level: str = "info") -> dict:
    """Record a status line and push it live to every registered listener.

    Returns the recorded ``{seq, text, level, ts}`` dict. Does NOT print — use
    :func:`notify` for the print-and-record path.
    """
    global _seq
    text = str(text)
    with _LOCK:
        _seq += 1
        item = {"seq": _seq, "text": text, "level": level, "ts": time.time()}
        _buffer.append(item)
        listeners = list(_listeners)
    for q in listeners:
        try:
            q.put({"type": "status_line", "seq": item["seq"],
                   "data": item["text"], "level": item["level"]})
        except Exception:  # noqa: BLE001 — a wedged listener must never break others
            pass
    return item


def notify(text: str, *, level: str = "info", echo: bool = True) -> dict:
    """Print *text* to the console (unless ``echo=False``) AND record it on the bus."""
    if echo:
        print(text)
    return emit(text, level=level)


def snapshot(since: int = 0) -> dict:
    """Buffered lines with ``seq > since`` (in order) plus the latest seq.

    The panel passes the highest seq it has already shown; this returns only the
    lines it is missing, so the on-connect / on-done fetch never double-renders a
    line already delivered live during a turn.
    """
    with _LOCK:
        items = [dict(i) for i in _buffer if i["seq"] > since]
        latest = _seq
    return {"seq": latest, "messages": items}


def register_live(q) -> None:
    """Register a queue to receive live ``status_line`` events (one per turn)."""
    with _LOCK:
        _listeners.add(q)


def unregister_live(q) -> None:
    with _LOCK:
        _listeners.discard(q)
