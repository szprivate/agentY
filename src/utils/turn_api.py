"""A turn, reduced to what a program that asked for it wants back.

The panel follows a turn as a stream of events and draws each one. A top-level
agent (Claude Code, any MCP client) that hands agentY a message wants the
opposite: ONE structured answer when the turn is over. :func:`summarize` folds the
events :mod:`src.utils.turn_bus` kept for a turn into that answer, and the routes
in ``agentY_server`` (``POST /agentY/turns``, ``GET /agentY/turns/<id>``) serve it.

Pure on purpose — it takes event dicts and returns a dict — so the folding rules
are tested without a server. :func:`install` registers a turn-bus observer that
keeps the finished summaries for a while after the bus has forgotten the turn (the
bus drops an ended turn's events after two minutes; a client that started a turn
with ``wait=False`` may come back much later).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

from src.utils import turn_bus

# How many finished summaries are kept (in memory; gone on a host restart).
KEEP_FINISHED = 100

# Canvas-patch ops that put a review in front of the user / take it away again.
_REVIEW_OPEN = frozenset({"review_collector", "review_text"})
_REVIEW_CLOSE = frozenset({"review_released"})
# Patches that are bookkeeping, not nodes placed or changed.
_PATCH_QUIET = frozenset({"flow_state", "prompt_version", "show_again"})

_BRIEF = 240


def _brief(value, limit: int = _BRIEF) -> str:
    """One short line of *value* — a tool's input or result."""
    if isinstance(value, str):
        text = value
    else:
        try:
            import json
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _patch_brief(ev: dict) -> dict:
    """What a canvas patch did, in a few scalars: op plus where, never the payload."""
    out: dict = {"op": ev.get("op") or "set_params"}
    for key in ("node_id", "node_ids", "hook_node_id", "mode", "state", "reason", "title"):
        if ev.get(key) not in (None, "", []):
            out[key] = ev[key] if not isinstance(ev[key], str) else _brief(ev[key], 120)
    if isinstance(ev.get("params"), dict):
        out["params"] = sorted(ev["params"])[:12]
    if isinstance(ev.get("ops"), list):
        out["edits"] = len(ev["ops"])
    return out


def summarize(events: list, *, request_id: str = "", thread_id: str = "",
              ended: bool = False, asking: bool = False, dropped: int = 0,
              started: float | None = None, finished: float | None = None) -> dict:
    """Fold a turn's events into the structured result.

    ``status`` is ``running``, ``waiting`` (the turn stopped to ask a question, or
    a review is open), ``done`` or ``error``.
    """
    reply: list[str] = []
    tools: "OrderedDict[str, dict]" = OrderedDict()
    files: list[dict] = []
    seen_files: set = set()
    canvas: list[dict] = []
    errors: list[str] = []
    notes: list[str] = []
    plan: list = []
    usage: dict | None = None
    ask: dict | None = None
    review_open = False
    saw_done = False
    synthesized = False

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "text":
            reply.append(str(ev.get("data") or ""))
        elif kind == "tool":
            tid = str(ev.get("id") or f"anon{len(tools)}")
            name = str(ev.get("name") or "tool").split("] ", 1)[-1]
            row = tools.setdefault(tid, {"id": tid, "name": name, "agent": ev.get("agent") or "",
                                         "status": "running"})
            if ev.get("phase") == "call":
                row["input"] = _brief(ev.get("input", ""))
            else:
                result = ev.get("result", "")
                row["result"] = _brief(result)
                row["status"] = "error" if str(result).lstrip().lower().startswith("error") else "ok"
        elif kind == "output":
            path = ev.get("path")
            if path and path not in seen_files:
                seen_files.add(path)
                files.append({"path": path, "kind": ev.get("kind") or "file",
                              "name": ev.get("name") or "", "role": ev.get("role") or ""})
        elif kind == "canvas_patch":
            op = ev.get("op") or ""
            if op in _REVIEW_OPEN:
                review_open = True
            elif op in _REVIEW_CLOSE:
                review_open = False
            if op not in _PATCH_QUIET and op not in _REVIEW_OPEN and op not in _REVIEW_CLOSE:
                canvas.append(_patch_brief(ev))
        elif kind == "ask":
            ask = {"kind": ev.get("kind") or "", "prompt": str(ev.get("prompt") or ""),
                   "request_id": ev.get("request_id") or request_id}
        elif kind == "usage":
            usage = {k: ev.get(k) for k in ("input", "output", "cache_read", "cache_write",
                                            "cache_hit", "cost", "calls") if k in ev}
        elif kind == "plan":
            plan = list(ev.get("steps") or [])
        elif kind == "error":
            errors.append(str(ev.get("message") or ""))
        elif kind == "system":
            text = str(ev.get("data") or "").strip()
            if text:
                notes.append(_brief(text))
        elif kind == "done":
            saw_done = True
            synthesized = bool(ev.get("synthesized"))

    ended = bool(ended or saw_done)
    if not asking:
        ask = None          # an old question the turn has since moved past
    waiting = None
    if ask:
        waiting = ask
    elif review_open:
        waiting = {"kind": "review", "prompt": "A review is open on the canvas: edit its "
                   "collector node to choose what continues, then send a message to resume."}
    if waiting:
        status = "waiting"
    elif not ended:
        status = "running"
    elif errors and not "".join(reply).strip():
        status = "error"
    else:
        status = "done"

    out = {
        "turn_id": request_id,
        "conversation": thread_id,
        "status": status,
        "finished": ended,
        "reply": "".join(reply).strip(),
        "tool_calls": list(tools.values()),
        "files": files,
        "canvas": canvas,
        "waiting": waiting,
        "errors": errors,
        "notes": notes[-5:],
        "plan": plan,
        "usage": usage,
    }
    if started:
        out["started"] = started
    if finished:
        out["seconds"] = round(finished - started, 1) if started else None
    if dropped:
        out["truncated_events"] = dropped
    if synthesized:
        out["synthesized_done"] = True
    return out


# ── Finished turns, kept past the bus's own memory ───────────────────────────

_lock = threading.Lock()
_finished: "OrderedDict[str, dict]" = OrderedDict()
_installed = False


def _observer(event: dict, turn) -> None:
    if event.get("type") != "done":
        return
    snap = turn_bus.snapshot(turn.request_id)
    if snap is None:
        return
    result = summarize(snap["events"], request_id=turn.request_id, thread_id=turn.thread_id,
                       ended=True, dropped=snap["dropped"], started=turn.started,
                       finished=snap["ended"] or time.time())
    result["origin"] = turn.origin
    result["message"] = _brief(turn.text, 200)
    with _lock:
        _finished[turn.request_id] = result
        while len(_finished) > KEEP_FINISHED:
            _finished.popitem(last=False)


def install() -> None:
    """Start keeping finished turns' summaries. Idempotent."""
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
    turn_bus.observe(_observer)


def finished(request_id: str) -> "dict | None":
    with _lock:
        row = _finished.get(str(request_id))
        return dict(row) if row else None


def current(request_id: str) -> "dict | None":
    """The summary of turn *request_id* as it stands: live from the bus, or the
    kept one after it ended. None when the id is unknown to both."""
    snap = turn_bus.snapshot(request_id)
    turn = turn_bus.turn(request_id)
    if snap is not None and turn is not None:
        out = summarize(snap["events"], request_id=turn.request_id, thread_id=turn.thread_id,
                        ended=bool(snap["ended"]), asking=snap["asking"],
                        dropped=snap["dropped"], started=turn.started,
                        finished=snap["ended"] or None)
        out["origin"] = turn.origin
        out["message"] = _brief(turn.text, 200)
        cur = turn_bus.current_tool(request_id)
        if cur:
            out["current_tool"] = cur
        return out
    return finished(request_id)


def recent_turns(limit: int = 30) -> list:
    """Short rows for the turns running now and those kept, newest first."""
    rows = []
    seen = set()
    for t in reversed(turn_bus.active()):
        seen.add(t.request_id)
        rows.append({"turn_id": t.request_id, "conversation": t.thread_id, "origin": t.origin,
                     "status": "waiting" if turn_bus.asking(t.request_id) else "running",
                     "message": _brief(t.text, 120), "started": t.started})
    with _lock:
        kept = list(_finished.values())
    for r in reversed(kept):
        if r["turn_id"] in seen:
            continue
        rows.append({"turn_id": r["turn_id"], "conversation": r["conversation"],
                     "origin": r.get("origin", ""), "status": r["status"],
                     "message": r.get("message", ""), "started": r.get("started")})
    return rows[:limit]
