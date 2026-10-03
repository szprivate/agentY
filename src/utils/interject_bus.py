"""Mid-run interjections: what the user says while a turn is already running.

The panel normally parks anything typed during a turn and sends it once the turn
ends (the ``⏳`` chips). An interjection skips that wait: ``POST /agentY/interject``
drops the text here, and the orchestrator's ``InterjectHookProvider`` picks it up
at the next tool boundary and hands it to the model.

One mailbox per running turn. Several conversations can run at once, each on its
own pipeline, so a message is addressed by the turn's request id and the reading
side — a hook running inside a turn — reads the mailbox of the turn it belongs to
(:mod:`agenty_core.utils.turn_scope`). ``open_run`` is called as the turn
registers, ``close_run`` when it ends — and close hands back anything that was
never delivered, so a message that arrived a moment too late goes back to the
panel's queue instead of vanishing.

Code outside a turn scope (a test, a single-turn host) reads the one open mailbox
when there is exactly one, which is what this module did before there could be
more.

Thread-safe on purpose: ``post`` runs on a Flask request thread while the drain
side runs inside the turn's own event loop, in another thread entirely.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
# request id -> {"thread": str, "pending": [{"text", "urgent"}], "relayed": [set]}.
# `relayed` runs parallel to `pending`: the specialists each message has already
# been SHOWN (see peek_for). Showing is not delivering — the orchestrator still
# drains it — so this lives beside the message rather than removing it.
_runs: dict[str, dict] = {}


def _own(req_id: str | None = None) -> dict | None:
    """The mailbox of *req_id*, else of the turn this code runs in. Caller holds _lock."""
    rid = str(req_id or "")
    if not rid:
        try:
            from agenty_core.utils import turn_scope
            rid = turn_scope.current().request_id
        except Exception:  # noqa: BLE001
            rid = ""
    if rid:
        return _runs.get(rid)
    return next(iter(_runs.values())) if len(_runs) == 1 else None


def open_run(req_id: str, thread_id: str = "") -> None:
    """Start accepting interjections for *req_id*.

    The thread id rides along so the delivering side can write the message into
    the conversation at the moment the model actually saw it — persisting on
    arrival instead would double up whatever comes back out of close_run.
    """
    if not req_id:
        return
    with _lock:
        _runs[str(req_id)] = {"thread": str(thread_id or ""), "pending": [], "relayed": []}


def thread_id(req_id: str | None = None) -> str:
    """Thread the run belongs to ('' when it is not running)."""
    with _lock:
        run = _own(req_id)
        return run["thread"] if run else ""


def close_run(req_id: str) -> list[str]:
    """End *req_id* and return the texts that were never delivered.

    A message posted after the agent's last tool call has nowhere left to land —
    the caller hands these back to the panel, which re-queues them as an ordinary
    next-turn message. Closing a run that is not open changes nothing.
    """
    with _lock:
        run = _runs.pop(str(req_id or ""), None)
        return [p["text"] for p in run["pending"]] if run else []


def active_run() -> str | None:
    """The request id of the turn this code runs in, if it is open — or, outside
    any turn, of the one open run when there is exactly one."""
    with _lock:
        run = _own()
        if run is None:
            return None
        return next((rid for rid, r in _runs.items() if r is run), None)


def active_runs() -> list[str]:
    with _lock:
        return list(_runs)


def post(req_id: str, text: str, urgent: bool = False) -> bool:
    """Queue an interjection for the running turn *req_id*. False if there is
    nothing to interject into (not running, a stale request id, or empty text)."""
    text = (text or "").strip()
    if not text:
        return False
    with _lock:
        run = _runs.get(str(req_id or ""))
        if run is None:
            return False
        run["pending"].append({"text": text, "urgent": bool(urgent)})
        run["relayed"].append(set())
        return True


def pending_count(req_id: str | None = None) -> int:
    with _lock:
        run = _own(req_id)
        return len(run["pending"]) if run else 0


def has_urgent(req_id: str | None = None) -> bool:
    with _lock:
        run = _own(req_id)
        return bool(run) and any(p["urgent"] for p in run["pending"])


def drain(req_id: str | None = None) -> list[dict]:
    """Take everything queued, in the order it was sent, and clear the mailbox."""
    with _lock:
        run = _own(req_id)
        if not run:
            return []
        out = list(run["pending"])
        run["pending"].clear()
        run["relayed"].clear()
        return out


def peek_for(listener: str, req_id: str | None = None) -> list[dict]:
    """Messages *listener* has not been shown yet — marked shown, NOT taken.

    For a specialist working inside one of the orchestrator's tool calls. It sees
    the message at its own next step, so the work in hand can change course, and
    the message stays in the mailbox for the orchestrator to read when the
    delegation returns. Each listener is shown each message once.
    """
    with _lock:
        run = _own(req_id)
        if not run:
            return []
        fresh = []
        for item, seen in zip(run["pending"], run["relayed"]):
            if listener not in seen:
                seen.add(listener)
                fresh.append(dict(item))
        return fresh


def drain_detailed(req_id: str | None = None) -> list[dict]:
    """:func:`drain`, with ``relayed_to``: the specialists already shown each message."""
    with _lock:
        run = _own(req_id)
        if not run:
            return []
        out = [{**item, "relayed_to": sorted(seen)}
               for item, seen in zip(run["pending"], run["relayed"])]
        run["pending"].clear()
        run["relayed"].clear()
        return out
