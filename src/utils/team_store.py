"""Who asked, what it cost, and what the team has told the agent to remember.

Three small things a shared server needs and a desktop does not:

* **Requesters.** A conversation started from a Slack channel (or by a user
  token) remembers who started it, in which channel, under which project. Asking
  "who may approve this?" and "whose turn was that?" both start here.
* **Usage ledger.** One JSON line per finished turn - user, channel/project,
  tokens, cost - appended to ``memory/team/usage.jsonl``. :func:`usage_summary`
  adds it up per user or per project.
* **Notes.** Plain markdown per user and per project (a Slack channel is a
  project). They are read into the turn that person starts, so "I always want
  16:9 and no text in the image" is said once. Written with ``remember`` in
  Slack, or by hand.

Plain files, no database: a team of a dozen is a few thousand lines a year.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger("agentY.team")

_ROOT = Path(__file__).resolve().parent.parent.parent
_lock = threading.RLock()
_requesters: dict = {}          # thread_id -> {"user", "channel", "project", "name"}
_last_usage: dict = {}          # request_id -> last usage event of that turn
_recorded: set = set()

NOTE_LIMIT = 6000               # characters kept per notes file when read into a turn


def team_dir() -> Path:
    return Path(os.environ.get("AGENTY_TEAM_DIR") or (_ROOT / "memory" / "team"))


# ── who asked ─────────────────────────────────────────────────────────────────
_loaded = False


def _requesters_file() -> Path:
    return team_dir() / "threads.json"


def _load_requesters() -> None:
    """Pick up who owns which conversation after a restart (once per process)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        data = json.loads(_requesters_file().read_text(encoding="utf-8"))
        for k, v in (data if isinstance(data, dict) else {}).items():
            if isinstance(v, dict):
                _requesters.setdefault(k, v)
    except (OSError, ValueError):
        pass


def _save_requesters() -> None:
    try:
        path = _requesters_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_requesters, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.exception("team: could not save the conversation owners")


def set_requester(thread_id: str, *, user: str, channel: str = "", project: str = "",
                  name: str = "") -> None:
    """The first person to speak in a conversation owns it. Later calls do not
    take it over - an approval has to stay with whoever asked."""
    thread_id = str(thread_id or "")
    if not thread_id or not user:
        return
    with _lock:
        _load_requesters()
        if thread_id in _requesters:
            return
        _requesters[thread_id] = {"user": str(user), "channel": str(channel),
                                  "project": str(project or channel), "name": name}
        _save_requesters()


def requester(thread_id: str) -> dict:
    with _lock:
        _load_requesters()
        return dict(_requesters.get(str(thread_id or ""), {}))


def forget_requester(thread_id: str) -> None:
    with _lock:
        _load_requesters()
        if _requesters.pop(str(thread_id or ""), None) is not None:
            _save_requesters()


# ── what it cost ──────────────────────────────────────────────────────────────
def usage_path() -> Path:
    return team_dir() / "usage.jsonl"


def on_turn_event(event: dict, turn) -> None:
    """Turn-bus observer: remember a turn's last usage, write the line at its end.

    Only turns somebody is attributed to are written - a panel turn on a
    desktop install leaves no trace here.
    """
    kind = str(event.get("type") or "")
    rid = getattr(turn, "request_id", "")
    if kind == "usage":
        with _lock:
            _last_usage[rid] = {k: event.get(k) for k in
                                ("input", "output", "cache_read", "cache_write",
                                 "cost", "partner_cost", "calls")}
        return
    if kind != "done":
        return
    with _lock:
        usage = _last_usage.pop(rid, {})
        _load_requesters()
        who = _requesters.get(str(turn.thread_id), {})
        if not who or rid in _recorded:
            return
        _recorded.add(rid)
        if len(_recorded) > 5000:
            _recorded.clear()
    record_usage(who, turn, usage)


def record_usage(who: dict, turn, usage: dict) -> None:
    line = {"ts": int(time.time()), "user": who.get("user", ""), "name": who.get("name", ""),
            "channel": who.get("channel", ""), "project": who.get("project", ""),
            "thread_id": getattr(turn, "thread_id", ""), "request_id": getattr(turn, "request_id", ""),
            "origin": getattr(turn, "origin", ""), "text": str(getattr(turn, "text", ""))[:120],
            "seconds": round(time.time() - getattr(turn, "started", time.time()), 1),
            **{k: (v or 0) for k, v in (usage or {}).items()}}
    try:
        path = usage_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("team: could not write the usage ledger")


def usage_summary(by: str = "user", since: float = 0.0, path: Path | None = None) -> list:
    """Totals per user or per project, biggest spender first."""
    key = "project" if by == "project" else "user"
    totals: dict = {}
    try:
        lines = (path or usage_path()).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if float(rec.get("ts") or 0) < since:
            continue
        t = totals.setdefault(str(rec.get(key) or "(none)"),
                              {key: str(rec.get(key) or "(none)"), "turns": 0, "input": 0,
                               "output": 0, "cost": 0.0, "partner_cost": 0.0})
        t["turns"] += 1
        for f in ("input", "output"):
            t[f] += int(rec.get(f) or 0)
        for f in ("cost", "partner_cost"):
            t[f] = round(t[f] + float(rec.get(f) or 0), 4)
    return sorted(totals.values(), key=lambda r: (-r["cost"], -r["turns"]))


# ── notes ─────────────────────────────────────────────────────────────────────
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _notes_file(scope: str, key: str) -> Path:
    if scope not in ("users", "projects"):
        raise ValueError("scope is 'users' or 'projects'")
    return team_dir() / scope / (_SAFE.sub("_", str(key)) + ".md")


def read_notes(scope: str, key: str) -> str:
    if not key:
        return ""
    try:
        return _notes_file(scope, key).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def add_note(scope: str, key: str, text: str) -> str:
    """Append one line of notes; returns the file's new content."""
    text = " ".join(str(text or "").split())
    if not key or not text:
        raise ValueError("nothing to remember")
    path = _notes_file(scope, key)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"- {text}\n")
        return path.read_text(encoding="utf-8").strip()


def clear_notes(scope: str, key: str) -> bool:
    try:
        _notes_file(scope, key).unlink()
        return True
    except OSError:
        return False


def context_block(who: dict) -> str:
    """The text put in front of a team member's request: who asked, and the notes."""
    if not who or not who.get("user"):
        return ""
    parts = [f"[REQUESTED BY {who.get('name') or who['user']} ({who['user']})"
             + (f" in {who['project']}" if who.get("project") else "") + "]"]
    mine = read_notes("users", who["user"])
    proj = read_notes("projects", who.get("project", ""))
    if mine:
        parts.append("Notes this person asked you to keep:\n" + mine[-NOTE_LIMIT:])
    if proj:
        parts.append("Notes for this project/channel:\n" + proj[-NOTE_LIMIT:])
    return "\n".join(parts)


def start_recording() -> None:
    from src.utils import turn_bus
    turn_bus.observe(on_turn_event)


def stop_recording() -> None:
    from src.utils import turn_bus
    turn_bus.unobserve(on_turn_event)
