"""Turn-lifecycle breadcrumbs, plus a stack dump when a turn stops making progress.

Motivation: the side panel sometimes goes quiet *after* a turn — the answer is on
screen, but no ``done`` ever arrives, so the panel stays in its streaming state,
queues anything you type, and only a host restart clears it. That failure is
invisible in ``message_history.log`` (which records model traffic, not the
plumbing around it), and by the time a human notices, the evidence — which thread
is parked on which call — is only in the live process.

So this module does two things, both cheap enough to leave on permanently:

* **Breadcrumbs.** Every turn walks a fixed sequence of phases (see
  ``_run_pipeline_stream``). Each transition appends one line to
  ``.logs/turn_trace.log``. A hung turn's last line names the phase it died in.
* **A stall dump.** A watchdog thread notices when a turn sits in one phase past
  that phase's budget and writes the stacks of *every* thread in the process to
  the same log, once per stall. That dump is the actual diagnosis: it names the
  exact frame the turn is parked on.

``snapshot()`` exposes the same view live over HTTP (``GET /agentY/diag``), so a
hang can be diagnosed in the session it happens in rather than reproduced.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / ".logs"
_LOG_PATH = _LOG_DIR / "turn_trace.log"

_lock = threading.Lock()
# req_id -> {"thread_id", "phase", "since", "started", "thread_name", "dumped"}
_turns: dict[str, dict] = {}
_watchdog: threading.Thread | None = None

# How long a phase may run before it counts as stalled. The model-streaming phase
# legitimately takes minutes (a video render blocks the turn), so it gets a long
# rope; everything after the last event is local bookkeeping that should take
# milliseconds, so a minute there is already pathological.
_BUDGETS = {
    "stream": 1800.0,
    "sse:open": 1800.0,
}
_DEFAULT_BUDGET = 60.0
_POLL = 5.0


def _write(line: str) -> None:
    stamped = f"{time.strftime('%H:%M:%S')} [{threading.current_thread().name}] {line}"
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break a turn
        pass
    if os.environ.get("AGENTY_TURN_TRACE_STDERR", "").strip() in ("1", "true", "yes", "on"):
        print(f"⏱ {stamped}", file=sys.stderr, flush=True)


def _dump_all_threads(header: str) -> str:
    """Render every live thread's stack. This is the payload of a stall report."""
    names = {t.ident: t.name for t in threading.enumerate()}
    out = [f"\n{'=' * 78}", header, "=" * 78]
    for ident, frame in sys._current_frames().items():
        out.append(f"\n--- thread {names.get(ident, '?')} ({ident}) ---")
        out.extend(x.rstrip() for x in traceback.format_stack(frame))
    out.append("=" * 78 + "\n")
    return "\n".join(out)


def _watch() -> None:
    while True:
        time.sleep(_POLL)
        now = time.monotonic()
        stalled = []
        with _lock:
            for req_id, st in _turns.items():
                if st.get("dumped"):
                    continue
                budget = _BUDGETS.get(st["phase"], _DEFAULT_BUDGET)
                held = now - st["since"]
                if held > budget:
                    st["dumped"] = True
                    stalled.append((req_id, dict(st), held))
        for req_id, st, held in stalled:
            _write(
                f"STALL req={req_id[:8]} phase={st['phase']} held={held:.1f}s "
                f"(budget {_BUDGETS.get(st['phase'], _DEFAULT_BUDGET):.0f}s) — dumping all threads"
            )
            _write(_dump_all_threads(
                f"STALL DUMP req={req_id[:8]} phase={st['phase']} held={held:.1f}s"
            ))


def _ensure_watchdog() -> None:
    global _watchdog
    if _watchdog is not None and _watchdog.is_alive():
        return
    _watchdog = threading.Thread(target=_watch, name="agentY-turn-watchdog", daemon=True)
    _watchdog.start()


def begin(req_id: str, thread_id: str = "") -> None:
    _ensure_watchdog()
    now = time.monotonic()
    with _lock:
        _turns[req_id] = {
            "thread_id": thread_id, "phase": "begin", "since": now, "started": now,
            "thread_name": threading.current_thread().name, "dumped": False,
        }
    _write(f"BEGIN req={req_id[:8]} thread={thread_id}")


def phase(req_id: str, name: str) -> None:
    """Record that *req_id* just entered phase *name*.

    Entering a new phase re-arms the watchdog: a turn that stalls twice in one
    run is reported twice, because the second stall is a different bug.
    """
    now = time.monotonic()
    with _lock:
        st = _turns.get(req_id)
        if st is None:
            return
        prev, held = st["phase"], now - st["since"]
        st.update(phase=name, since=now, dumped=False)
    _write(f"PHASE req={req_id[:8]} {prev} → {name} (prev took {held:.2f}s)")


def end(req_id: str, note: str = "") -> None:
    now = time.monotonic()
    with _lock:
        st = _turns.pop(req_id, None)
    if st is None:
        return
    _write(f"END   req={req_id[:8]} total={now - st['started']:.2f}s "
           f"last_phase={st['phase']} {note}".rstrip())


def note(req_id: str, msg: str) -> None:
    """A breadcrumb that doesn't change the phase (keep-alives, client disconnect)."""
    _write(f"NOTE  req={req_id[:8]} {msg}")


def is_in_flight(req_id: str) -> bool:
    """True while *req_id*'s runner is between ``begin`` and ``end``.

    The SSE generator uses this as a liveness check: a queue that has gone quiet
    while its runner is no longer tracked means the runner exited without
    terminating the stream, and waiting longer would hang the panel forever.
    """
    with _lock:
        return req_id in _turns


def snapshot(include_stacks: bool = True) -> dict:
    """Live view of in-flight turns (+ thread stacks) for ``GET /agentY/diag``."""
    now = time.monotonic()
    with _lock:
        turns = [
            {
                "request_id": rid, "thread_id": st["thread_id"], "phase": st["phase"],
                "in_phase_s": round(now - st["since"], 2),
                "total_s": round(now - st["started"], 2),
                "owner_thread": st["thread_name"],
                "over_budget": (now - st["since"]) > _BUDGETS.get(st["phase"], _DEFAULT_BUDGET),
            }
            for rid, st in _turns.items()
        ]
    data: dict = {
        "in_flight": turns,
        "threads": sorted(t.name for t in threading.enumerate()),
        "thread_count": threading.active_count(),
    }
    if include_stacks:
        data["stacks"] = _dump_all_threads("ON-DEMAND DUMP (GET /agentY/diag)")
    return data
