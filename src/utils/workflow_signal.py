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


class _State:
    """One turn's mailbox. Per turn (:mod:`agenty_core.utils.turn_scope`): two
    conversations running at once must not hand each other their workflows."""
    __slots__ = ("pending_paths", "dead_refused", "hold", "hold_fired")

    def __init__(self) -> None:
        self.pending_paths: list[str] = []
        # Paths already handed back once over nodes that would never execute (see
        # dead_nodes_refused_once). Cleared with the queue.
        self.dead_refused: set[str] = set()
        self.hold: dict | None = None
        self.hold_fired = False


def _st() -> _State:
    from agenty_core.utils import turn_scope
    return turn_scope.current().slot("workflow_signal", _State)


def set_execution_hold(payload: dict | None) -> None:
    """Refuse every ``signal_workflow_ready`` this turn, answering with *payload*.

    ``signal_workflow_ready`` is a module-level tool shared with the subagents, not
    a closure over the pipeline, so the one place both can see is this mailbox —
    the same bus the paths already travel on. Set at the start of a turn whose plan
    the user asked to approve, cleared at the end of it. ``None`` lifts the hold.
    """
    with _lock:
        _st().hold = dict(payload) if payload else None
        _st().hold_fired = False


def execution_hold() -> dict | None:
    """The refusal in force, or None when signalling is allowed."""
    with _lock:
        if not _st().hold:
            return None
        _st().hold_fired = True
        return dict(_st().hold)


def hold_fired() -> bool:
    """Whether the hold actually stopped something since it was set.

    The difference matters at the end of the turn: a held turn that never tried
    to run anything (a question, a chat) has not put a plan to the user, so it
    must not open the gate for the next one.
    """
    with _lock:
        return _st().hold_fired


def dead_nodes_refused_once(path: str) -> bool:
    """Whether this is the FIRST time *path* has been handed back over dead nodes.

    A workflow signalled with nodes ComfyUI would never execute is returned to the
    agent once, while it still has the turn and can wire or remove them. Once only:
    a second refusal of the same file would be a loop, and a graph that is merely
    carrying something useless must never become a run that never happens. So the
    second signal of the same path goes through, with the note that says what will
    not run.

    Lives here, next to the queue, because ``signal_workflow_ready`` is a
    module-level tool shared with the subagents rather than a closure over the
    pipeline — this mailbox is the one place they can all see. Cleared with the
    queue, so the memory lasts exactly one handoff.
    """
    with _lock:
        if path in _st().dead_refused:
            return False
        _st().dead_refused.add(path)
        return True


def append_workflow_path(path: str) -> None:
    """Append *path* to the pending queue (used for batch runs)."""
    with _lock:
        _st().pending_paths.append(path)


def set_workflow_path(path: str) -> None:
    """Store *path*, replacing any previously queued paths (single-workflow compat)."""
    with _lock:
        _st().pending_paths = [path]


def peek() -> list[str]:
    """The pending paths, left in place (for reporting what a stop would discard)."""
    with _lock:
        return list(_st().pending_paths)


def clear_and_get() -> list[str]:
    """Atomically read and clear all pending paths.

    Returns a list of workflow paths (empty list if none are queued).
    For a normal (non-batch) run the list contains exactly one entry.
    """
    with _lock:
        paths = list(_st().pending_paths)
        _st().pending_paths = []
        _st().dead_refused.clear()
        return paths
