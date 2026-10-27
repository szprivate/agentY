"""agentY's side of the partner-credit checks (the logic is agenty_core.partner_credits).

* the confirmation level comes from settings (``[partner_credits] confirm_above_usd``),
  read on every check so an edit in Settings applies to the next run;
* every measured spend is added to the turn's usage (``usage.partner_cost``) and
  shown beside the LLM cost;
* :func:`gate` is the look the tools take BEFORE a run starts, so the agent hears
  ``confirm_cost`` / ``insufficient_credits`` while it still has the turn - the
  executor's own gate stays as the backstop for every other path.
"""
from __future__ import annotations

import json
from pathlib import Path

_registered = False


def _threshold() -> float | None:
    try:
        from src.utils.settings import load_settings
        v = ((load_settings() or {}).get("partner_credits") or {}).get("confirm_above_usd")
        return None if v is None else float(v)
    except Exception:  # noqa: BLE001
        return None


def _on_spend(usd: float, info: dict) -> None:
    try:
        from src.utils import turn_usage
        turn_usage.add_partner(usd)
    except Exception:  # noqa: BLE001
        pass


def ensure_registered() -> None:
    """Idempotent; called wherever a submission can start."""
    global _registered
    if _registered:
        return
    from agenty_core import partner_credits as pc
    pc.set_threshold_provider(_threshold)
    pc.add_listener(_on_spend)
    _registered = True


def gate(paths: list[str], confirm: bool = False) -> dict | None:
    """Pre-flight for workflow files about to run. None = go ahead; otherwise the
    refusal as a tool-result dict (``error_kind`` ``confirm_cost`` or
    ``insufficient_credits``). With *confirm* the go-ahead is remembered for the
    files, so the later submission does not ask again. Blocking (a network read
    when the graph is paid) - call it from a thread."""
    ensure_registered()
    from agenty_core import partner_credits as pc
    try:
        est = pc.preflight_many([str(p) for p in paths], confirm=confirm)
    except pc.CostGateError as gate_err:
        return gate_err.payload()
    except Exception:  # noqa: BLE001 - a failed check must not stop a render
        return None
    if confirm and est.get("paid"):
        for p in paths:
            pc.confirm_path(str(p), float("inf"))
    return None


def estimate_of(paths: list[str]) -> dict | None:
    """``{"min","max"}`` USD for files that contain paid nodes, else None."""
    ensure_registered()
    from agenty_core import partner_credits as pc
    lo = hi = 0.0
    paid = unknown = False
    for p in paths:
        try:
            est = pc.estimate_workflow(json.loads(Path(p).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
        if est["paid"]:
            paid = True
            lo += est["min_usd"]
            hi += est.get("known_max_usd", est["max_usd"]) or 0.0
            unknown = unknown or bool(est.get("price_unknown"))
    if not paid:
        return None
    # An unpriced node: the top is unknown, never $0.
    return {"min": round(lo, 4), "max": None if unknown else round(hi, 4),
            **({"price_unknown": True} if unknown else {})}
