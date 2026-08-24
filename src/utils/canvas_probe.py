"""canvas_probe — ask the open ComfyUI page a question and wait for its answer.

Everything else that crosses this boundary goes ONE way. :mod:`canvas_patch`
pushes edits at the panel and never hears back; the panel posts the graph up with
each message and never asks anything. This is the first thing that needs a reply,
and neither of those channels can carry one.

Not over SSE, specifically. The canvas-patch drain lives *inside* the
orchestrator's ``stream_async`` loop, so it only flushes between agent events — a
tool that blocks waiting for an answer would hold the very loop that was supposed
to deliver its question. So the panel long-polls ``GET /agentY/canvas_probe``
instead, on its own connection, and answers on ``POST /agentY/canvas_probe/reply``.
Independent of the turn, and it works while the agent is mid-tool.

Two facts about the page make this worth the plumbing at all:

* Only the page can draw the graph. ``app.canvas.canvas`` is a real 2D canvas, so
  ``toDataURL()`` returns the user's OWN view — their node positions, their
  colours, their collapsed nodes — which is the thing worth sending. Re-rendering
  the JSON headlessly gets a different picture of the same graph.
* Only the page knows what else is open. ComfyUI keeps several workflows in tabs;
  the API prompt the panel posts is the ACTIVE one, and nothing in it says whether
  it was one of five.

Contract: :func:`request` blocks the calling thread until the panel replies or
*timeout* passes. Both are ordinary outcomes — a page that is closed, hidden, or
mid-reload simply never answers, and a tool that cannot get its picture must say
so rather than hang the turn.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any

_LOCK = threading.Lock()
# Waiters keyed by probe id: {"event": Event, "reply": dict|None, "request": dict}.
_PENDING: dict[str, dict] = {}
# Probes handed to the panel but not yet answered, so a poll that arrives during
# a reload does not re-serve one already in flight — and a genuine retry can.
_SERVED: dict[str, float] = {}
_RESERVE_SECONDS = 6.0


def _sweep(now: float) -> None:
    """Forget served-but-unanswered probes so a reloaded panel can pick them up."""
    for pid, at in list(_SERVED.items()):
        if now - at > _RESERVE_SECONDS or pid not in _PENDING:
            _SERVED.pop(pid, None)


def request(kind: str, payload: dict | None = None, timeout: float = 20.0) -> dict:
    """Ask the panel for *kind* and block until it answers.

    Returns the panel's reply dict, or ``{"error": …, "timeout": True}`` when
    nobody answered in time — which is what a closed tab looks like from here, and
    is a normal answer rather than a fault.
    """
    pid = uuid.uuid4().hex[:12]
    event = threading.Event()
    entry = {"event": event, "reply": None, "answered": False,
             "request": {"probe_id": pid, "kind": str(kind or ""),
                         "payload": dict(payload or {})}}
    with _LOCK:
        _PENDING[pid] = entry
    try:
        if not event.wait(timeout=max(0.5, float(timeout or 0))):
            return {
                "error": "the ComfyUI page did not answer in "
                         f"{timeout:.0f}s — it is probably closed, reloading, or "
                         "the agentY panel has not been opened in it.",
                "timeout": True,
            }
        return entry["reply"] if isinstance(entry["reply"], dict) else {}
    finally:
        with _LOCK:
            _PENDING.pop(pid, None)
            _SERVED.pop(pid, None)


def take() -> dict | None:
    """The next probe waiting to be answered, or ``None``.

    Called by the panel's long-poll. A probe already handed out is held back
    briefly (:data:`_RESERVE_SECONDS`) rather than removed: if the page reloads
    between being given the question and answering it, the waiter is still
    blocked, and the reservation lapsing is what lets the new page pick it up.
    """
    now = time.time()
    with _LOCK:
        _sweep(now)
        for pid, entry in _PENDING.items():
            # An ANSWERED probe stays in _PENDING until its waiter wakes up and
            # cleans up, which is a real window: without this check the next poll
            # is handed a question that already has its answer, and the probe
            # actually waiting behind it is never served at all.
            if entry["answered"] or pid in _SERVED:
                continue
            _SERVED[pid] = now
            return dict(entry["request"])
    return None


def reply(probe_id: str, data: dict) -> bool:
    """Deliver the panel's answer. False when nothing was waiting for it.

    False is the ordinary case for a late answer — the waiter timed out and gave
    up — so it is worth distinguishing from a bad id, but not worth raising over.
    """
    pid = str(probe_id or "")
    with _LOCK:
        entry = _PENDING.get(pid)
        if entry is None or entry["answered"]:
            return False            # timed out, or already answered by someone
        entry["reply"] = dict(data or {})
        entry["answered"] = True
        _SERVED.pop(pid, None)
    entry["event"].set()
    return True


def pending_count() -> int:
    """How many probes are waiting (for the poll's own idle bookkeeping)."""
    with _LOCK:
        return len(_PENDING)


def clear() -> None:
    """Release every waiter. For tests, and for a host shutting down."""
    with _LOCK:
        entries = list(_PENDING.values())
        _PENDING.clear()
        _SERVED.clear()
    for entry in entries:
        entry["reply"] = {"error": "cancelled"}
        entry["event"].set()


def describe_open_workflows(workflows: Any) -> str:
    """The ``[OPEN WORKFLOWS]`` block, or ``""`` when there is nothing to say.

    Silent for the ordinary single-tab case: one workflow open is what everybody
    assumes already, and a block saying so on every turn is rent paid for nothing.
    It speaks up when there are several, because then "the canvas" is ambiguous in
    a way the user cannot see from the agent's replies — the graph it was handed
    is the ACTIVE tab, and a request about "the other one" needs them to switch.
    """
    rows = [w for w in (workflows or []) if isinstance(w, dict)]
    if len(rows) < 2:
        return ""
    lines = ["[OPEN WORKFLOWS]",
             f"{len(rows)} workflows are open in ComfyUI's tabs. You were given "
             "the ACTIVE one:"]
    for w in rows:
        name = str(w.get("name") or w.get("filename") or "untitled").strip()
        marks = []
        if w.get("active"):
            marks.append("ACTIVE — this is the graph you have")
        if w.get("modified"):
            marks.append("unsaved changes")
        if w.get("temporary"):
            marks.append("never saved")
        nodes = w.get("nodes")
        if isinstance(nodes, int) and nodes >= 0 and w.get("active"):
            marks.append(f"{nodes} nodes")
        lines.append(f"  - {name}" + (f"  [{', '.join(marks)}]" if marks else ""))
    lines.append(
        "Only the active tab can be read, edited, run or screenshotted — ComfyUI "
        "keeps one graph in memory and the others as stored state. If the user "
        "means a different one, ask them to click that tab; do not try to open it "
        "yourself, which would replace what they are looking at.")
    return "\n".join(lines)
