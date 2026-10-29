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
    __slots__ = ("pending_paths", "dead_refused", "invalid_refused", "hold", "hold_fired",
                 "canvas_probe", "turn_started", "stale_refused")

    def __init__(self) -> None:
        self.pending_paths: list[str] = []
        # Paths already handed back once over nodes that would never execute (see
        # dead_nodes_refused_once). Cleared with the queue.
        self.dead_refused: set[str] = set()
        # How often each path was handed back for not validating (invalid_refusals).
        self.invalid_refused: dict[str, int] = {}
        self.hold: dict | None = None
        self.hold_fired = False
        # What this turn changed on the user's canvas (a callable the pipeline
        # sets), when the turn began, and the paths already handed back for
        # predating those changes (stale_against_canvas).
        self.canvas_probe = None
        self.turn_started = 0.0
        self.stale_refused: set[str] = set()


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


# A workflow that does not validate is handed back this often, then let through:
# a refusal that never ends is a run that never happens, and what ComfyUI then
# says about the graph goes to the repair that follows a failed run.
MAX_INVALID_REFUSALS = 2


def invalid_refusals(path: str) -> int:
    """Count one more hand-back of *path* for not validating; how many that makes."""
    with _lock:
        n = _st().invalid_refused.get(path, 0) + 1
        _st().invalid_refused[path] = n
        return n


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


def withdraw(path: str = "") -> list[str]:
    """Take *path* (every pending path when blank) back off this turn's queue.

    Signalled workflows reach ComfyUI only when the turn ends, so until then
    ComfyUI's own queue is empty and "nothing is queued" is false. Returns the
    paths removed.
    """
    want = str(path or "").strip()
    with _lock:
        st = _st()
        if not want:
            gone, st.pending_paths = list(st.pending_paths), []
            return gone
        from pathlib import Path as _P
        try:
            key = str(_P(want).resolve()).lower()
        except Exception:  # noqa: BLE001
            key = want.lower()
        gone = [p for p in st.pending_paths if p.lower() in (key, want.lower())]
        st.pending_paths = [p for p in st.pending_paths if p not in gone]
        return gone


def clear_and_get() -> list[str]:
    """Atomically read and clear all pending paths.

    Returns a list of workflow paths (empty list if none are queued).
    For a normal (non-batch) run the list contains exactly one entry.
    """
    with _lock:
        paths = list(_st().pending_paths)
        _st().pending_paths = []
        _st().dead_refused.clear()
        _st().invalid_refused.clear()
        return paths


def set_canvas_probe(probe, turn_started: float) -> None:
    """Tell ``signal_workflow_ready`` how to ask what this turn changed on the canvas.

    *probe* returns the list of canvas edits made this turn (empty when there were
    none, or on a turn whose edits are part of running it: hooks, prompt loop).
    """
    with _lock:
        _st().canvas_probe = probe
        _st().turn_started = float(turn_started or 0.0)
        _st().stale_refused = set()


def stale_against_canvas(path: str) -> list[str]:
    """The canvas edits *path* does not contain - said ONCE per path, else ``[]``.

    Asked to fix the workflow on the canvas, the orchestrator fixed it there
    (edit_canvas_graph) and then signalled the file it had built in an EARLIER
    turn. That file is not the canvas: the run would have been the old graph,
    and the fix was lost (CX15, night_20261029: the same DualCLIPLoader error
    three times after it had been replaced on the canvas). A file written during
    this turn is taken to be deliberate - a new build next to the edits.
    """
    import os
    with _lock:
        st = _st()
        probe, started = st.canvas_probe, st.turn_started
        if probe is None or not started or path in st.stale_refused:
            return []
    try:
        edits = list(probe() or [])
    except Exception:  # noqa: BLE001 - never stop a run on this
        return []
    if not edits:
        return []
    try:
        if os.path.getmtime(path) >= started:
            return []
    except OSError:
        return []
    with _lock:
        _st().stale_refused.add(path)
    return edits
