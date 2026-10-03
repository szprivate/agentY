"""Several conversations at once: one pipeline per running conversation.

A pipeline holds one turn's state on itself (the canvas it was handed, the plan
gate, the session's outputs …) and its agents refuse a second call while one is
in flight ("Agent is already processing a request"). So the host used to run one
turn at a time, and a second conversation had to wait for the first to finish.

The pool hands each running conversation a pipeline of its own. The first exists
from start; more are built when a conversation starts while every pipeline is
busy — about half a second each, since the expensive parts (models' clients, MCP
servers, caches) are shared — up to ``max_size``. A conversation already running
is not given a second pipeline: its turns stay in order. When every slot is
taken, a new turn waits for one to free up.

Pipelines are interchangeable: a turn loads its conversation into whichever one
it gets (``_restore_state``). The one that served the conversation last is
preferred, because its agents still hold that conversation's history and loading
it is then a no-op.

A settings change that rebuilds agents reaches idle pipelines at once and busy
ones as they are released (:meth:`PipelinePool.defer`).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


class ConversationBusy(RuntimeError):
    """The conversation already has a turn running."""


class PoolTimeout(RuntimeError):
    """No pipeline came free in time."""


class PipelinePool:
    def __init__(self, first: Any, factory: Callable[[], Any] | None = None,
                 max_size: "int | Callable[[], int]" = 5) -> None:
        self._all: list = [first] if first is not None else []
        self._factory = factory
        # A number, or a callable read each time a turn needs a pipeline — so a
        # changed setting applies to the next conversation, no restart.
        self._max_size = max_size
        self._busy: dict[int, str] = {}       # id(pipeline) -> thread it is running
        self._last: dict[int, str] = {}       # id(pipeline) -> thread it ran last
        self._deferred: dict[int, list] = {}  # id(pipeline) -> callbacks run at release
        self._building = 0
        self._cond = threading.Condition()

    @property
    def max_size(self) -> int:
        try:
            n = self._max_size() if callable(self._max_size) else self._max_size
            return max(1, int(n or 1))
        except Exception:  # noqa: BLE001
            return 1

    # ── what is there ──────────────────────────────────────────────────────
    @property
    def primary(self) -> Any:
        """The pipeline built at start (what single-pipeline callers mean)."""
        with self._cond:
            return self._all[0] if self._all else None

    def pipelines(self) -> list:
        with self._cond:
            return list(self._all)

    def idle(self) -> list:
        with self._cond:
            return [p for p in self._all if id(p) not in self._busy]

    def running_threads(self) -> list[str]:
        """The conversations with a turn running right now."""
        with self._cond:
            return [t for t in self._busy.values() if t]

    def is_running(self, thread_id: str) -> bool:
        with self._cond:
            return bool(thread_id) and thread_id in self._busy.values()

    def pipeline_of(self, thread_id: str) -> Any:
        """The pipeline running *thread_id*, or None."""
        with self._cond:
            for p in self._all:
                if self._busy.get(id(p)) == thread_id:
                    return p
        return None

    # ── taking and giving back ─────────────────────────────────────────────
    def acquire(self, thread_id: str, *, timeout: float | None = None,
                on_wait: Callable[[], None] | None = None,
                on_build: Callable[[], None] | None = None) -> Any:
        """A pipeline for *thread_id*'s turn, marked busy until :meth:`release`.

        Raises :class:`ConversationBusy` when that conversation already has a turn
        running, :class:`PoolTimeout` when none came free within *timeout*.
        *on_wait* is called once if the turn has to wait, *on_build* once if a
        pipeline is built for it — both so the caller can say why it is slow.
        """
        thread_id = str(thread_id or "")
        deadline = None if timeout is None else time.monotonic() + timeout
        waited = False
        with self._cond:
            while True:
                if thread_id and thread_id in self._busy.values():
                    raise ConversationBusy(thread_id)
                idle = [p for p in self._all if id(p) not in self._busy]
                if idle:
                    mine = [p for p in idle if self._last.get(id(p)) == thread_id]
                    pick = (mine or idle)[0]
                    self._busy[id(pick)] = thread_id
                    return pick
                if self._factory is not None and len(self._all) + self._building < self.max_size:
                    self._building += 1
                    break
                if not waited and on_wait is not None:
                    waited = True
                    try:
                        on_wait()
                    except Exception:  # noqa: BLE001
                        pass
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    raise PoolTimeout(f"no free agent within {timeout:.0f}s")
                self._cond.wait(timeout=left if left is not None else 5.0)
        # Built outside the lock: other conversations keep starting and ending.
        if on_build is not None:
            try:
                on_build()
            except Exception:  # noqa: BLE001
                pass
        try:
            pipeline = self._factory()
        except BaseException:
            with self._cond:
                self._building -= 1
                self._cond.notify_all()
            raise
        with self._cond:
            self._building -= 1
            self._all.append(pipeline)
            self._busy[id(pipeline)] = thread_id
            return pipeline

    def release(self, pipeline: Any) -> None:
        """Give *pipeline* back, after running what was deferred until it was free."""
        with self._cond:
            thread_id = self._busy.pop(id(pipeline), None)
            if thread_id is not None:
                self._last[id(pipeline)] = thread_id
            todo = self._deferred.pop(id(pipeline), [])
        for fn in todo:
            try:
                fn(pipeline)
            except Exception:  # noqa: BLE001
                pass
        with self._cond:
            self._cond.notify_all()

    # ── changes that must wait for a pipeline to be free ──────────────────
    def defer(self, fn: Callable[[Any], None]) -> dict:
        """Run ``fn(pipeline)`` on every pipeline: idle ones now, busy ones when
        they are released. Returns ``{"now": [...], "later": n}`` with what the
        idle ones returned."""
        with self._cond:
            idle = [p for p in self._all if id(p) not in self._busy]
            busy = [p for p in self._all if id(p) in self._busy]
            for p in busy:
                self._deferred.setdefault(id(p), []).append(fn)
            # Held while fn runs, so no turn starts on a pipeline being rebuilt.
            for p in idle:
                self._busy[id(p)] = ""
        now = []
        try:
            for p in idle:
                now.append(fn(p))
        finally:
            with self._cond:
                for p in idle:
                    if self._busy.get(id(p)) == "":
                        self._busy.pop(id(p), None)
                self._cond.notify_all()
        return {"now": now, "later": len(busy)}
