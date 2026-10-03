"""One conversation at a time edits the open canvas.

Several conversations can run at once (src/utils/pipeline_pool.py), and they all
see the same ComfyUI canvas. Each turn works from the graph the panel sent with
its message, so a second conversation editing it alongside would work from a
picture that is already out of date: it would not know the first one's new
nodes, could take their ids, and could edit them thinking they were its own.

So the first conversation to change the canvas holds it until its turn ends; the
others may read it, and are told to build a separate workflow (or wait) instead
of changing it. A turn that does not touch the canvas never takes the lease, so
conversations that only build and run workflows are not held up at all.
"""

from __future__ import annotations

import threading
from typing import Callable

_lock = threading.Lock()
_holder: str = ""
# Which conversations are running — set by the host. A holder that is no longer
# running cannot keep the canvas (a turn that died before its release).
_alive: Callable[[], list] | None = None


def set_alive_check(fn: Callable[[], list] | None) -> None:
    global _alive
    _alive = fn


def claim(thread_id: str) -> str:
    """Take the canvas for *thread_id*. Returns "" when it has it, else the
    conversation that does. Outside a conversation (CLI, tests) there is no lease."""
    global _holder
    thread_id = str(thread_id or "")
    if not thread_id:
        return ""
    with _lock:
        if _holder and _holder != thread_id and _alive is not None:
            try:
                if _holder not in (_alive() or []):
                    _holder = ""
            except Exception:  # noqa: BLE001
                pass
        if _holder and _holder != thread_id:
            return _holder
        _holder = thread_id
        return ""


def release(thread_id: str) -> None:
    """Give the canvas back, if *thread_id* has it."""
    global _holder
    with _lock:
        if _holder and _holder == str(thread_id or ""):
            _holder = ""


def holder() -> str:
    with _lock:
        return _holder
