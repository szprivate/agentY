"""Which browser panel a turn's canvas edits belong to.

A turn that nobody streams (the turn API, Slack, a shot) has no browser behind
it, so it asks one for the graph. It used to ask ANY: the host raised one global
"post me the graph" flag, whichever open panel answered first filled one global
cache, and the agent then planned (and addressed) its edits against THAT
browser's workflow. A script driving agentY through the turn API, with its own
headless page open, edited the user's own tab instead.

So every panel now names itself (a per-page client id, sent with its heartbeat,
its canvas snapshot and its messages), and says which conversation it shows. A
turn's canvas target is then one of

``"none"``          no canvas: canvas tools answer that none is attached;
``"conversation"``  a panel that is showing this turn's conversation (the default
                    for API and Slack turns);
``<client id>``     that one panel, provided it shows this turn's conversation
                    (two tabs on one conversation: pick which).

and anything else that is not an open panel resolves to nothing, never to
"whichever answered last". Panel-started turns do not go through here at all:
they carry their own graph with the message.

Stdlib only.
"""
from __future__ import annotations

import threading
import time

NONE = "none"
CONVERSATION = "conversation"

# The panel heartbeats every 5 s; a hidden tab's timers are throttled to about a
# minute. Longer than that and the panel is treated as closed.
PANEL_TTL = 75.0

_LOCK = threading.Lock()
_panels: dict[str, dict] = {}      # client id -> {"thread": str, "seen": float}


def touch(client: str, thread_id: str = "") -> None:
    """A panel is alive and shows *thread_id* (empty: the conversation is unknown)."""
    client = str(client or "").strip()[:80]
    if not client:
        return
    now = time.time()
    with _LOCK:
        entry = _panels.setdefault(client, {"thread": "", "seen": now})
        entry["seen"] = now
        if thread_id:
            entry["thread"] = str(thread_id)
        for cid in [c for c, e in _panels.items() if now - e["seen"] > 3600]:
            _panels.pop(cid, None)          # long gone: do not collect them forever


def alive(client: str) -> bool:
    with _LOCK:
        e = _panels.get(str(client or ""))
        return bool(e and time.time() - e["seen"] <= PANEL_TTL)


def thread_of(client: str) -> str:
    with _LOCK:
        return (_panels.get(str(client or "")) or {}).get("thread", "")


def normalize(target) -> str:
    """The request's ``canvas`` field as a target string; ``""`` means the default."""
    t = str(target or "").strip()
    if not t:
        return ""
    low = t.lower()
    return low if low in (NONE, CONVERSATION) else t


def resolve(target, thread_id: str) -> str:
    """The client id of the panel that *target* names for a turn in *thread_id*,
    or ``""`` when there is none (no panel shows that conversation, the named
    panel is closed or shows another, or the target is ``"none"``)."""
    target = normalize(target) or CONVERSATION
    if target == NONE or not thread_id:
        return ""
    now = time.time()
    with _LOCK:
        live = {c: e for c, e in _panels.items()
                if now - e["seen"] <= PANEL_TTL and e["thread"] == str(thread_id)}
    if target == CONVERSATION:
        return max(live, key=lambda c: live[c]["seen"]) if live else ""
    return target if target in live else ""


def why_none(target, thread_id: str) -> str:
    """One sentence for the agent: why this turn has no canvas to edit."""
    target = normalize(target) or CONVERSATION
    if target == NONE:
        return "the request asked for no canvas (canvas: \"none\")"
    if target == CONVERSATION:
        return "no ComfyUI panel is showing this conversation"
    if alive(target):
        return "the named panel is showing a different conversation"
    return "the named panel is not open"


def _reset_for_tests() -> None:
    with _LOCK:
        _panels.clear()
