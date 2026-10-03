"""A lead conversation and the shot conversations it starts.

A sequence is worked on by several agents at once: the user briefs one
conversation (the *lead*), it plans the sequence and starts a conversation per
shot. A shot is an ordinary conversation — its own agent, its own history, in the
conversation list with the 🟢 while it works, open-able and steerable by hand —
that happens to have been started, and briefed, by the lead.

When a shot's turn ends, the lead hears about it: the shot's report goes into the
lead's conversation and the lead gets a short turn of its own to review it (send a
correction, start the next shot, tell the user). Reports that arrive while the
lead is busy are collected and delivered together once it is free.

One level only: a shot cannot start shots of its own.

The host wires this up (:func:`configure`) with how to start a turn in a
conversation, whether one is running, how to stop one, and how to hand a message
to a running one. Kept free of Flask and of the pipeline so it can be tested on
its own.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from src.utils import conversation_store as cs
from src.utils import turn_bus

logger = logging.getLogger("agentY.shots")

# How many shots one lead may start, and how many times shots may wake it before
# the user has said anything again — a guard against two agents talking to each
# other forever, not a limit anyone planning a sequence should meet.
MAX_SHOTS_PER_LEAD = 60
MAX_WAKES_WITHOUT_USER = 40
# How much of a shot's report the lead is handed when it wakes (read_shot has it all).
REPORT_CHARS = 2500
# How long a wake waits for the lead's own turn to finish before trying again.
WAKE_WAIT = 600.0
# Past the setting `shots_max_tool_calls` a shot's turn is told to wrap up and
# report; this much further on it is stopped. Research has no natural end, and
# a shot no one is watching can otherwise search for as long as the host runs.
BUDGET_GRACE = 0.25

_LOCK = threading.Lock()
_hooks: dict[str, Callable | None] = {
    "start_turn": None,    # (thread_id, text, *, origin, dry_run) -> request_id
    "is_running": None,    # (thread_id) -> bool
    "stop_thread": None,   # (thread_id) -> bool
    "interject": None,     # (thread_id, text) -> bool
    "became_lead": None,   # (thread_id) -> None: its turn now runs on the Lead model
}
_pending: dict[str, list[dict]] = {}   # lead -> reports not yet delivered
_waking: set[str] = set()               # leads with a wake on its way
_wakes: dict[str, int] = {}             # lead -> wakes since the user last wrote
_turn_state: dict[str, str] = {}        # request_id -> "failed" | "stopped" | "over_budget"
_tool_calls: dict[str, int] = {}        # request_id -> tool calls so far (shot turns)
_observing = {"on": False}


def configure(*, start_turn: Callable, is_running: Callable,
              stop_thread: Callable | None = None,
              interject: Callable | None = None,
              became_lead: Callable | None = None) -> None:
    """Called once by the host. Also starts listening to turns."""
    _hooks.update(start_turn=start_turn, is_running=is_running,
                  stop_thread=stop_thread, interject=interject, became_lead=became_lead)
    if not _observing["on"]:
        turn_bus.observe(_on_event)
        _observing["on"] = True


def max_tool_calls() -> int:
    """Setting ``shots_max_tool_calls`` (0 = no limit)."""
    try:
        from src.utils.settings import load_settings
        return max(0, int(load_settings().get("shots_max_tool_calls", 80) or 0))
    except Exception:  # noqa: BLE001
        return 80


def dry_run_default() -> bool:
    """Setting ``shots_dry_run``: shots build their workflows and stop there."""
    try:
        from src.utils.settings import load_settings
        return bool(load_settings().get("shots_dry_run", False))
    except Exception:  # noqa: BLE001
        return False


# ── what a shot is told ──────────────────────────────────────────────────────

def _briefing_message(lead_id: str, name: str, briefing: str, dry_run: bool) -> str:
    lead_title = str((cs.get_thread(lead_id) or {}).get("title") or "the lead conversation")
    notes = cs.get_sequence_notes(lead_id).strip()
    parts = [
        f"[SHOT {name}] You are working on shot **{name}** of a sequence. The lead "
        f"conversation (\"{lead_title}\") planned the sequence and briefed you below; "
        "it reads your final answer when you finish.",
        "",
        briefing.strip(),
    ]
    if notes:
        parts += ["", "[SEQUENCE NOTES — the same for every shot; keep to them]", notes]
    parts += [
        "",
        "[HOW TO WORK AS A SHOT]",
        "- Build this shot as its own workflow (prepare_workflow); do not change the "
        "workflow open on the user's canvas — other shots are running at the same time.",
        "- Name your outputs after the shot.",
    ]
    if dry_run:
        parts.append("- Dry run: build and validate the workflow, but do not queue or render "
                     "it. The user renders once they have reviewed it.")
    parts.append("- End with a short report: what you made, the workflow and output paths, "
                 "and anything the lead must decide.")
    return "\n".join(parts)


# ── the lead's actions ───────────────────────────────────────────────────────

def _err(msg: str, **kw) -> dict:
    return {"ok": False, "error": msg, **kw}


def _find(lead_id: str, name: str) -> dict | None:
    want = str(name or "").strip().lower()
    for s in cs.shots_of(lead_id):
        if s["name"].strip().lower() == want or s["thread_id"] == name:
            return s
    return None


def start_shot(lead_id: str, name: str, briefing: str, *,
               dry_run: bool | None = None) -> dict:
    """Start a conversation for shot *name* with *briefing* as its first message."""
    name = str(name or "").strip()
    if not lead_id:
        return _err("No conversation is running this — start_shot works from a conversation.")
    if cs.shot_of(lead_id):
        return _err("This conversation is itself a shot; only the lead conversation starts "
                    "shots. Do this shot's work yourself.")
    if not name:
        return _err("Give the shot a name (e.g. 'sh010').")
    if not str(briefing or "").strip():
        return _err("Give the shot a briefing: what it must make, from what, and how.")
    if _hooks["start_turn"] is None:
        return _err("Shots are not available in this host.")
    existing = _find(lead_id, name)
    if existing:
        return _err(f"Shot '{name}' already exists — use message_shot to give it more to do.",
                    thread_id=existing["thread_id"])
    if len(cs.shots_of(lead_id)) >= MAX_SHOTS_PER_LEAD:
        return _err(f"This sequence already has {MAX_SHOTS_PER_LEAD} shots.")
    dry = dry_run_default() if dry_run is None else bool(dry_run)
    tid = cs.create_thread(title=name)
    cs.set_shot(tid, lead_id, name, status="queued")
    text = _briefing_message(lead_id, name, briefing, dry)
    cs.add_message(tid, "user", text)
    try:
        rid = _hooks["start_turn"](tid, text, origin="lead", dry_run=dry)
    except Exception as exc:  # noqa: BLE001
        cs.set_shot_status(tid, "failed")
        return _err(f"Could not start shot '{name}': {exc}", thread_id=tid)
    # This conversation is a lead from here on, the rest of this turn included.
    if _hooks.get("became_lead"):
        try:
            _hooks["became_lead"](lead_id)
        except Exception:  # noqa: BLE001
            logger.debug("could not switch %s to the lead model", lead_id, exc_info=True)
    return {"ok": True, "shot": name, "thread_id": tid, "request_id": rid,
            "dry_run": dry, "status": "started"}


def message_shot(lead_id: str, name: str, text: str) -> dict:
    """More for a shot to do: handed to its running turn, or a new turn of it."""
    shot = _find(lead_id, name)
    if not shot:
        return _err(f"No shot '{name}' in this sequence.", shots=[s["name"] for s in cs.shots_of(lead_id)])
    text = str(text or "").strip()
    if not text:
        return _err("Nothing to send.")
    tid = shot["thread_id"]
    message = f"[FROM THE LEAD] {text}"
    if _hooks["is_running"] and _hooks["is_running"](tid):
        if _hooks["interject"] and _hooks["interject"](tid, message):
            return {"ok": True, "shot": shot["name"], "delivered": "into its running turn"}
        return _err(f"Shot '{shot['name']}' is busy and could not be reached — try again "
                    "when it reports back.")
    cs.add_message(tid, "user", message)
    cs.set_shot_status(tid, "queued")
    rid = _hooks["start_turn"](tid, message, origin="lead", dry_run=dry_run_default())
    return {"ok": True, "shot": shot["name"], "delivered": "as a new turn", "request_id": rid}


def stop_shot(lead_id: str, name: str) -> dict:
    shot = _find(lead_id, name)
    if not shot:
        return _err(f"No shot '{name}' in this sequence.")
    stopped = bool(_hooks["stop_thread"] and _hooks["stop_thread"](shot["thread_id"]))
    return {"ok": True, "shot": shot["name"], "stopped": stopped}


def stop_all(lead_id: str) -> list[str]:
    """Stop every running shot of *lead_id* and drop reports not yet delivered.
    Returns the names stopped."""
    with _LOCK:
        _pending.pop(lead_id, None)
    out = []
    for s in cs.shots_of(lead_id):
        if _hooks["is_running"] and _hooks["is_running"](s["thread_id"]):
            if _hooks["stop_thread"] and _hooks["stop_thread"](s["thread_id"]):
                out.append(s["name"])
    return out


def _last_answer(thread_id: str) -> str:
    msgs = (cs.get_thread(thread_id) or {}).get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "assistant" and str(m.get("content") or "").strip():
            return str(m["content"]).strip()
    return ""


def _state(shot: dict, running: bool) -> str:
    """Where a shot stands: running only while a turn of it really is (a host
    restart ends one without anything getting to record that)."""
    if running:
        return "running"
    if shot["status"] == "queued" and time.time() - float(shot.get("updated_at") or 0) < 120:
        return "queued"                  # just started; its turn is on its way
    return "stopped" if shot["status"] in ("running", "queued") else shot["status"]


def status(lead_id: str) -> list[dict]:
    """Every shot of *lead_id*: name, conversation, state, start of its last report."""
    out = []
    for s in cs.shots_of(lead_id):
        running = bool(_hooks["is_running"] and _hooks["is_running"](s["thread_id"]))
        state = _state(s, running)
        out.append({"shot": s["name"], "thread_id": s["thread_id"], "status": state,
                    "report": _last_answer(s["thread_id"])[:300]})
    return out


def read_shot(lead_id: str, name: str) -> dict:
    """A shot's whole last report and the files it made."""
    shot = _find(lead_id, name)
    if not shot:
        return _err(f"No shot '{name}' in this sequence.")
    t = cs.get_thread(shot["thread_id"]) or {}
    running = bool(_hooks["is_running"] and _hooks["is_running"](shot["thread_id"]))
    return {"ok": True, "shot": shot["name"], "thread_id": shot["thread_id"],
            "status": _state(shot, running),
            "report": _last_answer(shot["thread_id"]),
            "outputs": [g.get("path") for g in (t.get("gallery") or [])][-20:]}


def set_notes(lead_id: str, notes: str) -> dict:
    if cs.shot_of(lead_id):
        return _err("Sequence notes belong to the lead conversation, not to a shot.")
    cs.set_sequence_notes(lead_id, notes)
    return {"ok": True, "chars": len(notes or "")}


# ── hearing back ─────────────────────────────────────────────────────────────

def _on_event(event: dict, turn) -> None:
    """Turn observer: notes how each turn ends, and reports a shot's end to its lead."""
    kind = event.get("type")
    rid = turn.request_id
    if kind == "turn_start":
        if turn.origin in ("panel", "slack"):
            # The user is talking to this conversation again: shots may wake it anew.
            with _LOCK:
                _wakes.pop(turn.thread_id, None)
        shot = cs.shot_of(turn.thread_id)
        if shot:
            cs.set_shot_status(turn.thread_id, "running")
            _tool_calls[rid] = 0
        return
    # The shot's own steps, not those of the specialists it delegates to (an
    # assembly makes dozens inside one prepare_workflow call).
    if (kind == "tool" and rid in _tool_calls and event.get("phase") == "call"
            and event.get("agent", "orchestrator") == "orchestrator"):
        _count_tool_call(rid, turn.thread_id)
        return
    if kind == "error":
        _turn_state[rid] = "failed"
        return
    if kind == "system" and str(event.get("data") or "").lstrip().startswith("⏹"):
        _turn_state[rid] = "stopped"
        return
    if kind != "done":
        return
    _tool_calls.pop(rid, None)
    ended = _turn_state.pop(rid, "done")
    if ended == "over_budget":
        ended = "stopped"
    shot = cs.shot_of(turn.thread_id)
    if shot:
        cs.set_shot_status(turn.thread_id, ended)
        over = _budget_notes.pop(rid, 0)
        with _LOCK:
            _pending.setdefault(shot["lead_id"], []).append(
                {"shot": shot["name"], "thread_id": turn.thread_id, "status": ended,
                 "note": (f"stopped after {over} tool calls without reporting (the shot "
                          "tool budget, setting shots_max_tool_calls)") if over else ""})
        _wake_soon(shot["lead_id"])
    else:
        # A lead's own turn ended: reports that came in meanwhile can go now.
        with _LOCK:
            waiting = bool(_pending.get(turn.thread_id))
        if waiting:
            _wake_soon(turn.thread_id)


def _count_tool_call(rid: str, thread_id: str) -> None:
    limit = max_tool_calls()
    _tool_calls[rid] = n = _tool_calls.get(rid, 0) + 1
    if not limit:
        return
    if n == limit and _hooks.get("interject"):
        _hooks["interject"](thread_id, (
            f"[TOOL BUDGET] This shot has made {limit} tool calls in this turn. Stop "
            "searching and building now: finish with what you have and write your report "
            "— what you made, what is missing, and what the lead must decide."))
    elif n >= limit + max(1, int(limit * BUDGET_GRACE)) and _turn_state.get(rid) != "over_budget":
        _turn_state[rid] = "over_budget"
        _budget_notes[rid] = n
        if _hooks.get("stop_thread"):
            threading.Thread(target=_hooks["stop_thread"], args=(thread_id,),
                             name="agentY-shot-budget", daemon=True).start()


_budget_notes: dict[str, int] = {}


def _wake_soon(lead_id: str) -> None:
    with _LOCK:
        if lead_id in _waking:
            return
        _waking.add(lead_id)
    threading.Thread(target=_wake, args=(lead_id,), name="agentY-shot-wake",
                     daemon=True).start()


def _wake(lead_id: str) -> None:
    """Give the lead a turn with every report that has come in, once it is free."""
    try:
        deadline = time.monotonic() + WAKE_WAIT
        # Also waits out the moment between a turn's `done` and its agent being
        # given back, when a new turn there would be refused as already running.
        time.sleep(0.5)
        while _hooks["is_running"] and _hooks["is_running"](lead_id):
            if time.monotonic() > deadline:
                return            # still busy; its own `done` wakes it again
            time.sleep(0.5)
        if cs.get_thread(lead_id) is None:
            with _LOCK:
                _pending.pop(lead_id, None)
            return
        with _LOCK:
            reports = _pending.pop(lead_id, [])
            n = _wakes.get(lead_id, 0) + (1 if reports else 0)
            _wakes[lead_id] = n
        if not reports:
            return
        if n > MAX_WAKES_WITHOUT_USER:
            if n == MAX_WAKES_WITHOUT_USER + 1:
                cs.add_message(lead_id, "assistant",
                               f"⏸ The shots have reported back {MAX_WAKES_WITHOUT_USER} times "
                               "without a word from you, so I have stopped reviewing them on "
                               "my own. Write here to pick it up again.")
            return
        text = _wake_message(reports)
        cs.add_message(lead_id, "user", text)
        _hooks["start_turn"](lead_id, text, origin="shots", dry_run=False)
    except Exception:  # noqa: BLE001 — a failed wake must never take anything else down
        logger.exception("could not wake lead %s", lead_id)
    finally:
        with _LOCK:
            _waking.discard(lead_id)
            again = bool(_pending.get(lead_id))
        if again and not (_hooks["is_running"] and _hooks["is_running"](lead_id)):
            _wake_soon(lead_id)


def _wake_message(reports: list[dict]) -> str:
    word = {"done": "finished", "failed": "FAILED", "stopped": "was stopped"}
    lines = ["[SHOTS REPORTING BACK] Review what came in. Send a shot a correction "
             "(message_shot), start the next shots, or tell the user where the sequence "
             "stands. Keep it short; the user can open any shot to look for themselves.", ""]
    for r in reports:
        report = _last_answer(r["thread_id"])
        if len(report) > REPORT_CHARS:
            report = report[:REPORT_CHARS] + " … (read_shot has the rest)"
        lines.append(f"## Shot {r['shot']} {word.get(r['status'], r['status'])}")
        if r.get("note"):
            lines.append(f"_{r['note']}._")
        lines.append(report or "(no report)")
        lines.append("")
    return "\n".join(lines).strip()


def _reset_for_tests() -> None:
    with _LOCK:
        _pending.clear()
        _waking.clear()
        _wakes.clear()
        _turn_state.clear()
        _tool_calls.clear()
        _budget_notes.clear()
