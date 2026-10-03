"""Every turn's event stream, offered to anyone who wants to watch it.

The sidebar is fed by a queue: :func:`agentY_server._run_pipeline_turn` pushes
event dicts (``text``, ``tool``, ``output``, ``ask``, ``done``, …) into it and the
SSE route drains them into the panel. That queue belongs to one HTTP request, so
anything else that wants to see a turn — a Slack bridge, a log, a metrics sink —
has no way in, and the turn is a *single* event stream that only one reader can
have.

This makes it many. :func:`tee` wraps the queue at the one place a turn starts,
so every ``put`` reaches the panel exactly as before AND is handed to each
registered observer. Nothing inside the turn changes, and nothing inside the turn
has to know an observer exists.

Two guarantees an observer can build on, because a watcher that misses the end
leaves whatever it was drawing stuck mid-turn forever:

* the panel is served **first**, always — an observer never delays it;
* every turn ends with exactly one ``{"type": "done"}``, synthesised here if the
  turn managed to close its queue without emitting one.

Observers are called on the turn's own thread, so they must be quick: hand the
event to your own queue and get out. An observer that raises is logged and
dropped from that turn's delivery, never propagated into the turn.

A turn's events are also kept while it runs (and briefly after), so a watcher
that arrives late — the panel opening a conversation a lead started, or one
reloaded mid-turn — can replay it from the start and follow it live
(:func:`follow`).

Kept stdlib-only so any module can import it without an import cycle.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger("agentY.turn_bus")

_LOCK = threading.Lock()
_observers: list = []
_active: dict = {}          # request_id -> Turn
_last = {"thread_id": ""}   # the most recent turn's thread
_logs: dict = {}            # request_id -> _Log (running turns, and ended ones briefly)

# How many events a turn keeps for a late watcher, and how long an ended turn's
# are kept: long enough for a panel that polls every few seconds to catch the end.
LOG_LIMIT = 20000
LOG_KEEP_AFTER_END = 120.0


class _Log:
    def __init__(self, turn=None) -> None:
        self.turn = turn
        self.events: list = []
        self.dropped = 0          # events trimmed off the front
        self.ended = 0.0          # time.time() the turn ended, 0 while running
        self.cond = threading.Condition()

    def add(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            if len(self.events) > LOG_LIMIT:
                cut = len(self.events) - LOG_LIMIT
                del self.events[:cut]
                self.dropped += cut
            self.cond.notify_all()

    def end(self) -> None:
        with self.cond:
            if not self.ended:
                self.ended = time.time()
            self.cond.notify_all()


@dataclass(frozen=True)
class Turn:
    """What a turn IS, for a watcher that did not start it.

    ``origin`` is who asked: ``"panel"`` for the ComfyUI sidebar, ``"slack"`` for
    the Slack bridge, and whatever a future caller names itself. A mirror uses it
    to tell "you asked this" from "this is happening elsewhere" — the second is
    the whole reason to watch.
    """
    request_id: str
    thread_id: str
    origin: str = "panel"
    text: str = ""
    started: float = field(default_factory=time.time)


def observe(fn) -> None:
    """Register ``fn(event: dict, turn: Turn)``, called for every turn's events."""
    with _LOCK:
        if fn not in _observers:
            _observers.append(fn)


def unobserve(fn) -> None:
    with _LOCK:
        if fn in _observers:
            _observers.remove(fn)


def watching() -> bool:
    """Whether anyone is watching at all — lets a caller skip work nobody wants."""
    with _LOCK:
        return bool(_observers)


def active() -> list:
    """The turns currently in flight, newest last."""
    with _LOCK:
        return sorted(_active.values(), key=lambda t: t.started)


def last_thread_id() -> str:
    """The thread of the most recent turn, or "" before anything has run.

    "Where is the user right now?" — asked by a second channel that has to land
    its message in the conversation the person is actually having. The most
    recent turn is the honest answer and needs nothing reported from the panel.
    """
    with _LOCK:
        return _last["thread_id"]


def _publish(event: dict, turn: Turn) -> None:
    with _LOCK:
        observers = list(_observers)
    for fn in observers:
        try:
            fn(event, turn)
        except Exception:  # noqa: BLE001 — a watcher must never break the turn
            logger.exception("turn observer %r failed on %s", fn, event.get("type"))


class _Tee:
    """A queue that also tells the observers. Everything else is the real queue."""

    def __init__(self, inner, turn: Turn):
        self._inner = inner
        self._turn = turn
        self._done = False

    def put(self, item, *args, **kwargs):
        # The panel first, always: an observer that is slow, wedged or broken
        # must not show up as the agent being slow.
        self._inner.put(item, *args, **kwargs)
        log = _logs.get(self._turn.request_id)
        if isinstance(item, dict):
            if item.get("type") == "done":
                self._done = True
            if log is not None:
                log.add(item)
            _publish(item, self._turn)
        elif item is None:
            # `None` ends the stream. A watcher that never hears the end leaves
            # whatever it was drawing stuck mid-turn, so one is made up here if
            # the turn closed its queue without emitting one.
            if not self._done:
                self._done = True
                if log is not None:
                    log.add({"type": "done", "synthesized": True})
                _publish({"type": "done", "synthesized": True}, self._turn)
            if log is not None:
                log.end()
            _finish(self._turn)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def tee(inner, *, request_id: str, thread_id: str, origin: str = "panel",
        text: str = ""):
    """Wrap a turn's SSE queue so its events reach the observers too.

    Returns the wrapper to use in place of *inner* for the rest of the turn. Safe
    to call with no observers registered: the wrapper is then a thin pass-through
    that still records the turn as active (which is what a second channel checks
    before deciding whether to start one of its own).
    """
    turn = Turn(request_id=str(request_id), thread_id=str(thread_id),
                origin=str(origin or "panel"), text=str(text or ""))
    now = time.time()
    with _LOCK:
        _active[turn.request_id] = turn
        _last["thread_id"] = turn.thread_id
        for rid in [r for r, lg in _logs.items()
                    if lg.ended and now - lg.ended > LOG_KEEP_AFTER_END]:
            _logs.pop(rid, None)
        _logs[turn.request_id] = _Log(turn)
    _publish({"type": "turn_start"}, turn)
    return _Tee(inner, turn)


def _finish(turn: Turn) -> None:
    with _LOCK:
        _active.pop(turn.request_id, None)


def turn(request_id: str) -> "Turn | None":
    """Turn *request_id* — running, or ended moments ago — or None."""
    with _LOCK:
        t = _active.get(str(request_id))
        if t is None and str(request_id) in _logs:
            t = _logs[str(request_id)].turn
        return t


def follow(request_id: str, *, tick: float = 15.0):
    """A turn's events from its start, then live as they come, ending after `done`.

    Yields each event dict, and ``None`` every *tick* seconds of silence (for the
    caller's keep-alive). Yields nothing at all for a turn it has no record of.
    Events trimmed off a very long turn are announced with one ``system`` line.
    """
    with _LOCK:
        log = _logs.get(str(request_id))
    if log is None:
        return
    i = 0
    if log.dropped:
        yield {"type": "system", "data": "… earlier output of this turn is not shown."}
    while True:
        with log.cond:
            pos = i - log.dropped
            if pos < 0:
                i, pos = log.dropped, 0
            if pos >= len(log.events) and not log.ended:
                log.cond.wait(timeout=tick)
            batch = log.events[i - log.dropped:]
            ended = bool(log.ended)
        if not batch:
            if ended:
                return
            yield None
            continue
        for ev in batch:
            i += 1
            yield ev
            if isinstance(ev, dict) and ev.get("type") == "done":
                return


def recent(within: float = LOG_KEEP_AFTER_END) -> list:
    """Turns that ended in the last *within* seconds, as ``(Turn, ended_at)``.

    A turn that starts and ends between two of the panel's polls is otherwise
    never seen by a panel that has its conversation open (a lead woken briefly
    by its shots): this is how it finds out, and :func:`follow` still has it.
    """
    now = time.time()
    with _LOCK:
        return [(lg.turn, lg.ended) for lg in _logs.values()
                if lg.turn is not None and lg.ended and now - lg.ended <= within]
