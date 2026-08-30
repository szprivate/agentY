"""Technical QA requirements that are settled by measurement, not by judgement.

Half of what people put in a QA briefing is not a matter of opinion. *16:9. At
least 1080p. Not a soft render. No blown highlights.* Every one of those is
decided by a number :mod:`src.utils.image_facts` already computes, and asking a
vision model to rule on them instead is worse in three separate ways: it can be
wrong, it costs a round trip, and it is unrepeatable — the same image can pass on
Tuesday and fail on Wednesday.

So these are checked here, in code, with certainty. The model is told the answers
and told not to re-judge them, and spends its attention on the half that actually
needs it: whether the picture is any good.

This is the "hard gate" half of QA. A requirement here is not weighed against
anything — an output that is the wrong shape is the wrong shape, however
beautiful. Everything soft and negotiable stays in the written criteria.

The spec is whatever the `agentY qa briefing` node put on the canvas, so every key
is optional and an unknown one is ignored rather than fatal: a graph saved by a
newer node must still run here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("agentY.qa_checks")

# A ratio is a float, and a render is allowed to be a rounding away from its
# nominal shape: 1312x736 is 1.7826 where 16:9 is 1.7778, and calling that a
# failure would fail almost every real output.
RATIO_TOLERANCE = 0.02

RATIOS = {
    "16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4,
    "3:2": 3 / 2, "2:3": 2 / 3, "21:9": 21 / 9, "2.39:1": 2.39,
}

# The short side, which is how everyone actually means "1080p".
HEIGHTS = {"720p": 720, "1080p": 1080, "1440p": 1440, "2160p (4K)": 2160}

# Bands that satisfy each demand. Named rather than numeric so the thresholds
# stay in one place (image_facts) and this stays about policy.
SHARP_OK = {"sharp", "very sharp"}
SOFT_OK = {"very soft", "soft", "sharp", "very sharp"}     # anything goes
CLEAN_OK = {"clean", "light grain"}

# Above this fraction of pinned pixels, "no blown highlights" is not satisfied.
# Not zero: a spec highlight or a light source legitimately clips.
CLIP_LIMIT = 0.02


def _result(criterion: str, ok: bool, note: str) -> dict:
    """One verdict row, in the shape the QA agent's own checks already use."""
    return {"criterion": criterion, "result": "pass" if ok else "fail", "note": note}


def _ratio_check(spec: str, facts: dict) -> dict | None:
    want = RATIOS.get(spec)
    w, h = facts.get("width"), facts.get("height")
    if want is None or not (w and h):
        return None
    got = w / h
    ok = abs(got - want) <= RATIO_TOLERANCE * want
    return _result(f"aspect ratio {spec}", ok,
                   f"measured {w}x{h} = {got:.3f}; {spec} is {want:.3f}"
                   + ("" if ok else " — wrong shape"))


def _resolution_check(spec: str, facts: dict) -> dict | None:
    want = HEIGHTS.get(spec)
    w, h = facts.get("width"), facts.get("height")
    if want is None or not (w and h):
        return None
    short = min(w, h)
    ok = short >= want
    return _result(f"at least {spec}", ok,
                   f"measured {w}x{h}, short side {short}px"
                   + ("" if ok else f" — below {want}px"))


def _sharpness_check(spec: str, facts: dict) -> dict | None:
    sharp = (facts.get("sharpness") or {})
    band = sharp.get("band")
    if spec != "must be sharp" or not band:
        return None
    # A shallow depth of field reads soft overall while part of the frame is
    # genuinely sharp. Failing that would reject exactly the picture the brief
    # usually wants, so the sharpest region satisfies the demand on its own.
    ok = band in SHARP_OK or sharp.get("sharpest_band") in SHARP_OK
    note = f"sharpness {sharp.get('score')} ({band})"
    if ok and band not in SHARP_OK:
        note += (f", but sharpest region {sharp.get('sharpest_region')} "
                 f"({sharp.get('sharpest_band')}) — soft areas read as depth of field")
    elif not ok:
        note += " — the whole frame is soft"
    return _result("must be sharp, not a soft render", ok, note)


def _grain_check(spec: str, facts: dict) -> dict | None:
    ns = (facts.get("noise") or {})
    band = ns.get("band")
    if spec != "must be clean" or not band:
        return None
    ok = band in CLEAN_OK
    return _result("must be clean, not grainy", ok,
                   f"noise sigma {ns.get('sigma')} ({band})"
                   + ("" if ok else " — visible grain"))


def _clipping_check(enabled, facts: dict) -> dict | None:
    ex = (facts.get("exposure") or {})
    if not enabled or not ex:
        return None
    white = ex.get("clipped_white", 0.0)
    black = ex.get("clipped_black", 0.0)
    ok = white <= CLIP_LIMIT and black <= CLIP_LIMIT
    return _result("no blown highlights or crushed blacks", ok,
                   f"{white * 100:.1f}% blown, {black * 100:.1f}% crushed"
                   + ("" if ok else " — detail lost there is not recoverable"))


def _black_frames_check(enabled, facts: dict) -> dict | None:
    if not enabled or "black_frames" not in facts:
        return None
    n = facts.get("black_frames") or 0
    return _result("no black frames", n == 0,
                   f"{n} of {facts.get('frames_sampled', '?')} sampled frames are black")


def _motion_check(enabled, facts: dict) -> dict | None:
    if not enabled or "frozen_pairs" not in facts:
        return None
    n = facts.get("frozen_pairs") or 0
    return _result("the clip must not stall", n == 0,
                   f"{n} sampled pair(s) essentially identical"
                   if n else "motion throughout")


_CHECKS = {
    "aspect_ratio": _ratio_check,
    "resolution": _resolution_check,
    "sharpness": _sharpness_check,
    "grain": _grain_check,
    "no_clipping": _clipping_check,
    "no_black_frames": _black_frames_check,
    "no_stalled_motion": _motion_check,
}

# What the node writes when a control is left alone. Nothing is checked for these.
_UNSET = ("", "any", "off", None, False)


def evaluate(spec: dict, facts: dict) -> list[dict]:
    """Every technical requirement in *spec*, judged against measured *facts*.

    Returns verdict rows in the same shape the QA agent produces, so they merge
    into one list of checks and the user cannot tell — nor need to — which half
    was settled by arithmetic.

    A requirement whose fact is missing yields nothing at all rather than a
    failure: an unreadable file is not evidence that the output is wrong, and
    QA's whole disposition is that doubt does not condemn.
    """
    # No guard on empty facts: every check below already yields nothing without
    # the fact it needs, so an unmeasurable file falls through to no verdicts on
    # its own. A second guard here would be a branch no test could justify.
    if not isinstance(spec, dict):
        return []
    out: list[dict] = []
    for key, fn in _CHECKS.items():
        value = spec.get(key)
        if value in _UNSET:
            continue
        try:
            row = fn(value, facts)
        except Exception as exc:  # noqa: BLE001 — one bad key must not stop QA
            logger.debug("qa_checks: %s failed — %s", key, exc)
            continue
        if row:
            out.append(row)
    return out


def describe(spec: dict) -> str:
    """The technical requirements as lines for the briefing's criteria text.

    They go in the written criteria as well as being checked here, because the
    briefing is also what the user reads back — and a requirement that appears
    nowhere in it looks like it was ignored.
    """
    if not isinstance(spec, dict):
        return ""
    lines: list[str] = []
    if spec.get("aspect_ratio") not in _UNSET:
        lines.append(f"- aspect ratio {spec['aspect_ratio']}")
    if spec.get("resolution") not in _UNSET:
        lines.append(f"- at least {spec['resolution']}")
    if spec.get("sharpness") not in _UNSET:
        lines.append(f"- {spec['sharpness']}")
    if spec.get("grain") not in _UNSET:
        lines.append(f"- {spec['grain']}")
    if spec.get("no_clipping") not in _UNSET:
        lines.append("- no blown highlights or crushed blacks")
    if spec.get("no_black_frames") not in _UNSET:
        lines.append("- no black frames")
    if spec.get("no_stalled_motion") not in _UNSET:
        lines.append("- the clip must not stall")
    return "\n".join(lines)


def render_for_model(rows: list[dict]) -> str:
    """The settled results, told to the model so it does not re-judge them.

    Without this the model is shown an image and a criterion it has no way to
    measure, and guesses — which is the failure the whole module exists to
    remove.
    """
    if not rows:
        return ""
    lines = ["These technical criteria are ALREADY DECIDED by measuring the file. "
             "Do not re-judge them and do not contradict them; they are merged into "
             "your verdict for you:"]
    for r in rows:
        lines.append(f"- {r['criterion']}: {r['result'].upper()} ({r['note']})")
    return "\n".join(lines)
