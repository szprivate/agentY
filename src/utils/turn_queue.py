"""A line for turns asked from outside the panel (Slack, the turn API).

Why a queue at all, when the host can run conversations in parallel
(:mod:`src.utils.pipeline_pool`)? Because "can" is not "should, for everyone who
walks up". A pipeline per running conversation is real isolation - a turn's
buffers live in its own :mod:`turn_scope` - so two turns do not corrupt each
other. What they share is the *providers*: every running turn makes LLM calls,
and a team in a channel can ask five things in the same minute. The pool, once
full, makes a new turn wait silently (the panel shows a system line); a person
in Slack sees nothing.

This puts the limit in front of the pool, makes it a number you choose
(``server.max_running_turns``), and tells whoever is waiting where they stand.
First in, first out; a conversation that is already waiting takes a follow-up
message into the same turn instead of a second place in line.

Pure bookkeeping: the thing that actually starts a turn is a callable the
submitter hands over, so the queue is testable without a pipeline.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("agentY.turn_queue")

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
CANCELLED = "cancelled"

_ids = itertools.count(1)


class QueueFull(RuntimeError):
    """More turns are waiting than ``max_queued`` allows."""


@dataclass
class Ticket:
    id: int
    starter: object                 # callable() -> request_id
    text: str = ""
    user: str = ""
    thread_id: str = ""
    label: str = ""                 # where it came from, for the queue listing
    state: str = QUEUED
    request_id: str = ""
    enqueued: float = field(default_factory=time.time)
    started: float = 0.0
    ended: float = 0.0
    extra: dict = field(default_factory=dict)


class TurnQueue:
    def __init__(self, *, limit, max_queued=20, on_change=None):
        """*limit* is a number or a callable returning one, read at every decision
        so a settings change applies to the next turn without a restart."""
        self._limit = limit
        self._max_queued = max_queued
        self._listeners: list = [on_change] if on_change else []
        self._lock = threading.RLock()
        self._waiting: list = []                 # Tickets, FIFO
        self._running: dict = {}                 # ticket id -> Ticket
        self._by_request: dict = {}              # request_id -> Ticket

    # ── numbers ───────────────────────────────────────────────────────────────
    def limit(self) -> int:
        try:
            n = self._limit() if callable(self._limit) else self._limit
            return max(1, int(n))
        except Exception:  # noqa: BLE001
            return 1

    def max_queued(self) -> int:
        try:
            n = self._max_queued() if callable(self._max_queued) else self._max_queued
            return max(0, int(n))
        except Exception:  # noqa: BLE001
            return 0

    # ── in ────────────────────────────────────────────────────────────────────
    def submit(self, starter, *, text: str = "", user: str = "", thread_id: str = "",
               label: str = "", extra: dict | None = None) -> Ticket:
        """Take a place in line; starts at once when a slot is free.

        Raises :class:`QueueFull` when the line is at its limit.
        """
        t = Ticket(id=next(_ids), starter=starter, text=text, user=user,
                   thread_id=str(thread_id or ""), label=label, extra=dict(extra or {}))
        with self._lock:
            if len(self._waiting) >= self.max_queued() and len(self._running) >= self.limit():
                raise QueueFull(f"{len(self._waiting)} turns are already waiting")
            self._waiting.append(t)
        self._notify(t)
        self.pump()
        return t

    def amend(self, thread_id: str, text: str) -> Ticket | None:
        """Add *text* to a turn of this conversation that is still waiting."""
        thread_id = str(thread_id or "")
        if not thread_id:
            return None
        with self._lock:
            for t in self._waiting:
                if t.thread_id == thread_id:
                    t.text = (t.text + "\n\n" + text).strip()
                    return t
        return None

    def waiting_in(self, thread_id: str) -> Ticket | None:
        thread_id = str(thread_id or "")
        with self._lock:
            return next((t for t in self._waiting if thread_id and t.thread_id == thread_id), None)

    def cancel(self, ticket_id: int) -> bool:
        with self._lock:
            for t in self._waiting:
                if t.id == ticket_id:
                    self._waiting.remove(t)
                    t.state, t.ended = CANCELLED, time.time()
                    break
            else:
                return False
        self._notify(t)
        self._positions_changed()
        return True

    # ── the line moving ───────────────────────────────────────────────────────
    def pump(self) -> None:
        """Start waiting turns while there is room."""
        while True:
            with self._lock:
                if not self._waiting or len(self._running) >= self.limit():
                    return
                t = self._waiting.pop(0)
                t.state, t.started = RUNNING, time.time()
                self._running[t.id] = t
            try:
                rid = t.starter()
                t.request_id = str(rid or "")
                with self._lock:
                    if t.request_id:
                        self._by_request[t.request_id] = t
                if not t.request_id:
                    # Nothing was started (a command answered in place, say).
                    self._finish(t)
                    continue
            except Exception:  # noqa: BLE001 - one bad turn must not jam the line
                logger.exception("turn_queue: starting a turn failed")
                self._finish(t, failed=True)
                continue
            self._notify(t)
            self._positions_changed()

    def finished(self, request_id: str) -> None:
        """The turn *request_id* ended (its ``done`` event)."""
        with self._lock:
            t = self._by_request.pop(str(request_id), None)
        if t is not None:
            self._finish(t)
            self.pump()

    def _finish(self, t: Ticket, failed: bool = False) -> None:
        with self._lock:
            self._running.pop(t.id, None)
            t.state = CANCELLED if failed else DONE
            t.ended = time.time()
        self._notify(t)

    def sweep(self, alive_request_ids, grace: float = 30.0) -> None:
        """Release tickets whose turn is gone without a ``done`` ever arriving."""
        alive = set(alive_request_ids)
        now = time.time()
        with self._lock:
            lost = [t for t in self._running.values()
                    if t.request_id and t.request_id not in alive
                    and now - t.started > grace]
        for t in lost:
            self.finished(t.request_id)

    # ── looking ───────────────────────────────────────────────────────────────
    def position(self, ticket: Ticket) -> int:
        """1 = next to start; 0 = running or over."""
        with self._lock:
            try:
                return self._waiting.index(ticket) + 1
            except ValueError:
                return 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "limit": self.limit(),
                "running": [self._view(t) for t in self._running.values()],
                "queued": [dict(self._view(t), position=i + 1)
                           for i, t in enumerate(self._waiting)],
            }

    @staticmethod
    def _view(t: Ticket) -> dict:
        return {"id": t.id, "user": t.user, "thread_id": t.thread_id, "label": t.label,
                "request_id": t.request_id, "state": t.state,
                "waited_s": round((t.started or time.time()) - t.enqueued, 1),
                "text": t.text[:80]}

    # ── telling whoever listens ───────────────────────────────────────────────
    def add_listener(self, fn) -> None:
        """``fn(ticket)`` is called whenever a ticket's state or place changes."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, t: Ticket) -> None:
        for fn in list(self._listeners):
            try:
                fn(t)
            except Exception:  # noqa: BLE001
                logger.exception("turn_queue: a listener failed")

    def _positions_changed(self) -> None:
        with self._lock:
            rest = list(self._waiting)
        for t in rest:
            self._notify(t)


# ── the one line this host has ────────────────────────────────────────────────
_SHARED: "TurnQueue | None" = None
_shared_lock = threading.Lock()


def _setting_int(key: str, default: int) -> int:
    try:
        from src.utils.settings import load_settings
        return int((load_settings().get("server") or {}).get(key, default) or default)
    except Exception:  # noqa: BLE001
        return default


def _running_limit() -> int:
    """``server.max_running_turns``, never more than the pool can run."""
    n = _setting_int("max_running_turns", 2)
    try:
        from src.utils.settings import load_settings
        n = min(n, int(load_settings().get("parallel_chats", 5) or 5))
    except Exception:  # noqa: BLE001
        pass
    return max(1, n)


def shared() -> TurnQueue:
    """The host's queue. Slack and the turn API both wait in it, so one limit
    covers everything that is not somebody sitting at the panel."""
    global _SHARED
    with _shared_lock:
        if _SHARED is None:
            _SHARED = TurnQueue(limit=_running_limit,
                                max_queued=lambda: _setting_int("max_queued_turns", 20))
            try:
                from src.utils import turn_bus
                q = _SHARED

                def _done(event, turn):  # noqa: ANN001
                    if str(event.get("type")) == "done":
                        q.finished(turn.request_id)

                turn_bus.observe(_done)
            except Exception:  # noqa: BLE001
                logger.exception("turn_queue: could not watch for finished turns")
        return _SHARED
