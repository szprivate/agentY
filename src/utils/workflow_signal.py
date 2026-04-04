"""
workflow_signal – Thread-safe mailbox for the workflow path the Brain hands off.

The Brain calls ``signal_workflow_ready(workflow_path)`` as its very last step
instead of ``submit_prompt``.  The pipeline reads ``clear_and_get()`` after the
Brain finishes and passes the path to the Executor.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_pending_path: str | None = None


def set_workflow_path(path: str) -> None:
    """Store *path* so the pipeline can pick it up after the Brain exits."""
    global _pending_path
    with _lock:
        _pending_path = path


def clear_and_get() -> str | None:
    """Atomically read and clear the pending path.  Returns ``None`` if not set."""
    global _pending_path
    with _lock:
        path = _pending_path
        _pending_path = None
        return path
