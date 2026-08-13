"""Mid-run interjections: what the user says while a turn is already running.

The panel normally parks anything typed during a turn and sends it once the turn
ends (the ``⏳`` chips). An interjection skips that wait: ``POST /agentY/interject``
drops the text here, and the orchestrator's ``InterjectHookProvider`` picks it up
at the next tool boundary and hands it to the model.

One run is active at a time, deliberately. ``Pipeline`` keeps this turn's state on
itself (``_canvas_hooks``, ``_canvas_base_prompt``, ``_hook_run_stopped`` …), so a
second concurrent turn would corrupt the first; the panel refuses to start one and
so does this bus. ``open_run`` is called as the turn registers, ``close_run`` when
it ends — and close hands back anything that was never delivered, so a message
that arrived a moment too late goes back to the panel's queue instead of vanishing.

Thread-safe on purpose: ``post`` runs on a Flask request thread while the drain
side runs inside the turn's own event loop, in another thread entirely.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_active_run: str | None = None
_active_thread: str = ""
_pending: list[dict] = []


def open_run(req_id: str, thread_id: str = "") -> None:
    """Start accepting interjections for *req_id* (anything older is dropped).

    The thread id rides along so the delivering side can write the message into
    the conversation at the moment the model actually saw it — persisting on
    arrival instead would double up whatever comes back out of close_run.
    """
    global _active_run, _active_thread
    with _lock:
        _active_run = str(req_id) if req_id else None
        _active_thread = str(thread_id or "")
        _pending.clear()


def thread_id() -> str:
    """Thread the active run belongs to ('' when nothing is running)."""
    with _lock:
        return _active_thread


def close_run(req_id: str) -> list[str]:
    """End *req_id* and return the texts that were never delivered.

    A message posted after the agent's last tool call has nowhere left to land —
    the caller hands these back to the panel, which re-queues them as an ordinary
    next-turn message. Closing a run that is not the active one changes nothing.
    """
    global _active_run, _active_thread
    with _lock:
        if _active_run is not None and str(req_id) != _active_run:
            return []
        left = [p["text"] for p in _pending]
        _pending.clear()
        _active_run = None
        _active_thread = ""
        return left


def active_run() -> str | None:
    with _lock:
        return _active_run


def post(req_id: str, text: str, urgent: bool = False) -> bool:
    """Queue an interjection for the running turn. False if there is nothing to
    interject (no active run, a stale request id, or empty text)."""
    text = (text or "").strip()
    if not text:
        return False
    with _lock:
        if _active_run is None or str(req_id) != _active_run:
            return False
        _pending.append({"text": text, "urgent": bool(urgent)})
        return True


def pending_count() -> int:
    with _lock:
        return len(_pending)


def has_urgent() -> bool:
    with _lock:
        return any(p["urgent"] for p in _pending)


def drain() -> list[dict]:
    """Take everything queued, in the order it was sent, and clear the mailbox."""
    with _lock:
        out = list(_pending)
        _pending.clear()
        return out
