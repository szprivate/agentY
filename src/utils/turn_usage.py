"""What a turn has spent so far, for the panel's live usage line.

Token counts used to reach the user in two places only: the terminal (one line
per tool call) and the token-usage viewer (the log, afterwards). While a turn
runs, the panel said nothing about what it was costing.

Every agent's token hook reports here what it used since it last reported. The
total is kept per turn (:mod:`agenty_core.utils.turn_scope`), so with several
conversations running at once each one counts its own, and the specialists a
turn calls are counted with it - they run inside the same scope.

The stream that forwards a turn's events asks :func:`take_changed` on its short
timer and sends a ``usage`` event whenever the numbers moved.
"""
from __future__ import annotations

from agenty_core.utils import turn_scope

_SLOT = "turn_usage"


def _blank() -> dict:
    return {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
            "cost": 0.0, "calls": 0, "priced": False, "_sent": -1, "_rev": 0}


def _state(scope) -> dict:
    return scope.slot(_SLOT, _blank)


def add(d_in: int = 0, d_out: int = 0, d_cache_read: int = 0, d_cache_write: int = 0,
        d_cost: float | None = None, calls: int = 1) -> None:
    """Count what an agent used since its last report, in the current turn.

    *d_cost* None means the model has no known price: the tokens are counted and
    the cost is left as it was. Negative numbers (an accumulator that started
    over) are ignored rather than subtracted.
    """
    scope = turn_scope.current()
    with scope.lock:
        s = _state(scope)
        s["input"] += max(0, int(d_in or 0))
        s["output"] += max(0, int(d_out or 0))
        s["cache_read"] += max(0, int(d_cache_read or 0))
        s["cache_write"] += max(0, int(d_cache_write or 0))
        s["calls"] += max(0, int(calls or 0))
        if d_cost is not None:
            s["cost"] += max(0.0, float(d_cost))
            s["priced"] = True
        s["_rev"] += 1


def snapshot(scope=None) -> dict:
    """The turn's totals as the panel shows them."""
    scope = scope or turn_scope.current()
    with scope.lock:
        s = _state(scope)
        total_in = s["input"]
        return {
            "input": total_in,
            "output": s["output"],
            "cache_read": s["cache_read"],
            "cache_write": s["cache_write"],
            # share of the input that was read from the provider's cache
            "cache_hit": round(s["cache_read"] / total_in, 3) if total_in else 0.0,
            "cost": round(s["cost"], 4) if s["priced"] else None,
            "calls": s["calls"],
        }


def take_changed(scope=None) -> dict | None:
    """The totals, once per change: None when nothing was added since the last
    call (or nothing at all yet)."""
    scope = scope or turn_scope.current()
    with scope.lock:
        s = _state(scope)
        if s["_rev"] == 0 or s["_sent"] == s["_rev"]:
            return None
        if not (s["input"] or s["output"]):
            # The first model call has started and nothing is counted yet: a line
            # reading "0 in, 0 out" says nothing. Wait for the first numbers.
            return None
        s["_sent"] = s["_rev"]
        return snapshot(scope)
