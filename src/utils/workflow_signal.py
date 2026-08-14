"""
workflow_signal – Thread-safe mailbox for the workflow path(s) the Assemble Workflow hands off.

The Assemble Workflow calls ``signal_workflow_ready(workflow_path)`` as its very last step
instead of ``submit_prompt``.  For batch runs the Brain calls it once per
workflow file (each append adds to the queue).  The pipeline reads
``clear_and_get()`` after the Assemble Workflow finishes and receives the full list,
then passes each path to the Executor in sequence.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_pending_paths: list[str] = []
_hold: dict | None = None
_hold_fired: bool = False


def set_execution_hold(payload: dict | None) -> None:
    """Refuse every ``signal_workflow_ready`` this turn, answering with *payload*.

    ``signal_workflow_ready`` is a module-level tool shared with the subagents, not
    a closure over the pipeline, so the one place both can see is this mailbox —
    the same bus the paths already travel on. Set at the start of a turn whose plan
    the user asked to approve, cleared at the end of it. ``None`` lifts the hold.
    """
    global _hold, _hold_fired
    with _lock:
        _hold = dict(payload) if payload else None
        _hold_fired = False


def execution_hold() -> dict | None:
    """The refusal in force, or None when signalling is allowed."""
    global _hold_fired
    with _lock:
        if not _hold:
            return None
        _hold_fired = True
        return dict(_hold)


def hold_fired() -> bool:
    """Whether the hold actually stopped something since it was set.

    The difference matters at the end of the turn: a held turn that never tried
    to run anything (a question, a chat) has not put a plan to the user, so it
    must not open the gate for the next one.
    """
    with _lock:
        return _hold_fired


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


def peek() -> list[str]:
    """The pending paths, left in place (for reporting what a stop would discard)."""
    with _lock:
        return list(_pending_paths)


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
