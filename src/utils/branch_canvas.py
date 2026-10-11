"""A branch's own writes to the canvas, and their way onto it.

A lead starts its branches in conversations of their own, and they run at the
same time. They all look at one canvas. While each of them wrote to it directly,
the first to write held the canvas (src/utils/canvas_lease.py) and the others
were refused until its turn ended - so two branches took turns instead of
running side by side. And what a branch wrote never arrived: a branch's turn has
no page of its own to carry a patch, so the value was gone by its next turn.

So a branch does not write to the canvas at all:

* a VALUE it sets (a prompt, a seed, a size) is kept here, as the branch's own,
  and laid over the canvas it is handed at the start of each of its turns;
* the same write is queued for the lead, whose next turn puts it on the canvas
  the user is looking at;
* a change to the STRUCTURE of the graph is the lead's to make
  (see ``Pipeline._canvas_lease_refusal``).

Both lists live in the conversation store, so they survive a restart.
"""

from __future__ import annotations

import threading

_VALUES = "branch_values"
_OUTBOX = "canvas_outbox"
_MAX = 200
_lock = threading.Lock()


def is_value(patch: dict) -> bool:
    """A patch that sets values, as opposed to one that changes the graph."""
    if not isinstance(patch, dict):
        return False
    op = patch.get("op")
    return (op is None and isinstance(patch.get("params"), dict)) or op == "place_text"


def _key(patch: dict) -> tuple:
    if patch.get("op") == "place_text":
        return ("text", str(patch.get("hook_node_id")))
    return ("node", str(patch.get("node_id")))


def merged(patches: list, patch: dict) -> list:
    """*patches* with *patch* added: a later write to the same place replaces the
    earlier one (a node's parameters are joined, so each keeps its last value)."""
    out, done = [], False
    for old in patches:
        if not done and _key(old) == _key(patch):
            new = dict(old, **patch)
            if isinstance(old.get("params"), dict) and isinstance(patch.get("params"), dict):
                new["params"] = dict(old["params"], **patch["params"])
            out.append(new)
            done = True
        else:
            out.append(old)
    if not done:
        out.append(dict(patch))
    return out[-_MAX:]


def _read(thread_id: str, key: str) -> list:
    from src.utils import conversation_store as cs
    got = cs.kv_get(thread_id, key, [])
    return [p for p in got if isinstance(p, dict)] if isinstance(got, list) else []


def _write(thread_id: str, key: str, patches: list) -> None:
    from src.utils import conversation_store as cs
    cs.kv_set(thread_id, key, patches)


def note(thread_id: str, patch: dict) -> None:
    """Remember a value the branch *thread_id* set, for its own later turns."""
    if not thread_id or not is_value(patch):
        return
    with _lock:
        _write(thread_id, _VALUES, merged(_read(thread_id, _VALUES), patch))


def values(thread_id: str) -> list:
    """Every value the branch has set, oldest first."""
    return _read(thread_id, _VALUES) if thread_id else []


def overlay(prompt: dict, thread_id: str) -> int:
    """Lay the branch's own values over the canvas it was handed (in place).

    Returns how many nodes took a value. A node the user has since deleted, and
    an input they have since wired, are left alone.
    """
    if not isinstance(prompt, dict) or not prompt:
        return 0
    from src.utils.canvas_edit import apply_patch
    n = 0
    for patch in values(thread_id):
        if patch.get("op") is None and str(patch.get("node_id")) in prompt:
            apply_patch(prompt, patch)
            n += 1
    return n


def queue(lead_id: str, patch: dict, branch: str = "") -> None:
    """Hand a branch's write to its lead, to be put on the canvas."""
    if not lead_id or not is_value(patch):
        return
    with _lock:
        _write(lead_id, _OUTBOX,
               merged(_read(lead_id, _OUTBOX), dict(patch, from_branch=str(branch or "branch"))))


def take(lead_id: str) -> list:
    """The writes waiting for *lead_id*'s canvas - and they wait no longer."""
    if not lead_id:
        return []
    with _lock:
        waiting = _read(lead_id, _OUTBOX)
        if waiting:
            _write(lead_id, _OUTBOX, [])
    return waiting


def waiting(lead_id: str) -> int:
    return len(_read(lead_id, _OUTBOX)) if lead_id else 0
