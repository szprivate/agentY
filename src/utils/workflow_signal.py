"""
workflow_signal – Thread-safe mailbox for the workflow path(s) the Brain hands off.

The Brain calls ``signal_workflow_ready(workflow_path)`` as its very last step
instead of ``submit_prompt``.  For batch runs the Brain calls it once per
workflow file (each append adds to the queue).  The pipeline reads
``clear_and_get()`` after the Brain finishes and receives the full list,
then passes each path to the Executor in sequence.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_pending_paths: list[str] = []


def append_workflow_path(path: str) -> None:
    """Append *path* to the pending queue (used for batch runs)."""
    global _pending_paths
    with _lock:
        _pending_paths.append(path)


def set_workflow_path(path: str) -> None:
    """Store *path*, replacing any previously queued paths (single-workflow compat)."""
    global _pending_paths
    with _lock:
        _pending_paths = [path]


def clear_and_get() -> list[str]:
    """Atomically read and clear all pending paths.

    Returns a list of workflow paths (empty list if none are queued).
    For a normal (non-batch) run the list contains exactly one entry.
    """
    global _pending_paths
    with _lock:
        paths = list(_pending_paths)
        _pending_paths = []
        return paths
