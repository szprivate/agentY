"""Did a canvas edit actually land on a canvas?

A canvas patch used to be fire-and-forget: the tool pushed it into the turn's
event stream and answered "applied" straight away, whether or not any browser
ever drew it. A turn nobody was streaming (the turn API) reported dozens of
successful edits that no page had received.

Now a patch pushed for a turn that is bound to a page (``canvas_client`` on the
turn scope: that page's JS acknowledges what it applied) carries a ``patch_id``;
the page POSTs ``/agentY/canvas_ack`` when it has applied the patch, and the
tool waits for that before it says "applied". No acknowledgement in time: the
tool answers "not applied to any canvas", and the patch is marked expired so a
page that catches up later does not apply an edit the agent was told failed.

A turn whose page cannot acknowledge (an older panel: no client id) is not
tracked, and behaves as before.
"""
from __future__ import annotations

import threading
import time
import uuid

_LOCK = threading.Lock()
_pending: dict = {}      # patch id -> {"event", "ack"|None, "expired"}
_EXPIRED: dict = {}      # patch id -> time it expired (kept briefly, for late pages)


def track(patch: dict) -> str:
    """Give *patch* an id and expect an acknowledgement for it."""
    pid = uuid.uuid4().hex[:12]
    patch["patch_id"] = pid
    with _LOCK:
        if len(_pending) > 2000:
            _pending.clear()
        _pending[pid] = {"event": threading.Event(), "ack": None}
    return pid


def ack(pid: str, ok: bool, **detail) -> bool:
    """The page's answer. False when nobody waits any more."""
    with _LOCK:
        rec = _pending.get(str(pid))
        if rec is None or rec["ack"] is not None:
            return False
        rec["ack"] = {"ok": bool(ok), **detail}
    rec["event"].set()
    return True


def wait(pid: str, timeout: float) -> dict:
    """The page's answer for *pid*, or ``{"ok": False, "timeout": True}``."""
    with _LOCK:
        rec = _pending.get(str(pid))
    if rec is None:
        return {"ok": True, "untracked": True}
    got = rec["event"].wait(max(0.0, timeout))
    with _LOCK:
        _pending.pop(str(pid), None)
        if not got:
            now = time.time()
            for k in [k for k, t in _EXPIRED.items() if now - t > 600]:
                _EXPIRED.pop(k, None)
            _EXPIRED[str(pid)] = now
    return rec["ack"] if got and rec["ack"] else {"ok": False, "timeout": True}


def expired(pid) -> bool:
    """Was *pid* given up on? A page must not apply it then."""
    with _LOCK:
        return str(pid or "") in _EXPIRED


def _reset_for_tests() -> None:
    with _LOCK:
        _pending.clear()
        _EXPIRED.clear()
