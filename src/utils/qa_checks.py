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
import re

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

# What the likeness dropdown offers, and which scorer answers each option.
WANT_FACE = "must match the reference face"
WANT_SUBJECT = "must match the reference subject"
LIKENESS_SCORERS = {WANT_FACE: "face_match", WANT_SUBJECT: "subject_match"}

# Bands from src.utils.likeness that satisfy the demand. The face bar sits at
# "likely the same person" (0.70 cosine) because that lands in a gap rather than
# on a judgement call: on this machine's own renders the same character never
# scored below 0.95 and different characters never above 0.54.
FACE_OK = {"likely the same person", "the same person"}
SUBJECT_OK = {"the same subject", "clearly related"}


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


def _likeness_check(spec: str, facts: dict) -> dict | None:
    """Does the output actually look like the reference it was given?

    Yields nothing when the comparison could not be made — no reference with a
    face in it, no face in the output, or the scorer is not installed. The
    written criterion still reaches the model in that case, which is the right
    fallback: an eye answers this question less exactly, but it always can.
    """
    key = LIKENESS_SCORERS.get(spec)
    if key is None:
        return None
    m = facts.get(key) or {}
    if not m.get("available"):
        return None
    ok = m.get("band") in (FACE_OK if key == "face_match" else SUBJECT_OK)
    what = "face" if key == "face_match" else "subject"
    return _result(spec, ok,
                   f"{what} similarity {m.get('score')} vs {m.get('reference')} "
                   f"({m.get('band')})"
                   + ("" if ok else " — not the reference"))


_CHECKS = {
    "aspect_ratio": _ratio_check,
    "resolution": _resolution_check,
    "sharpness": _sharpness_check,
    "grain": _grain_check,
    "no_clipping": _clipping_check,
    "no_black_frames": _black_frames_check,
    "no_stalled_motion": _motion_check,
    "likeness": _likeness_check,
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


# ── a requirement stated in prose ───────────────────────────────────────────────
#
# The dropdowns on the `agentY qa briefing` node are the exact way to ask for a
# shape, and they are the only way `qa_repair` could ever hear about one: it is
# handed `technical`, and prose never reached it. So a briefing that said "16:9"
# in words was judged (the ratio is measured either way and shown to the model)
# and could not be repaired — the retry rerolled the seed, rewrote the prompt,
# and came back failing with the identical number.
#
# This reads the same requirement out of the words. Deliberately narrow: it fires
# on statements that are already unambiguous — a ratio, a pixel size, a named
# resolution — and stays silent on everything else. "Cinematic", "widescreen
# feel", "portrait orientation" are moods and orientations, not specs, and
# quietly rewriting somebody's render off one of those is worse than not firing.

# `:` only, deliberately. "16x9" is a legitimate way to write a ratio, but so is
# "a 2x3 grid of variations", and the two are indistinguishable at this size —
# reading the second as a 2:3 render is the kind of false positive that silently
# reshapes someone's output. Pixel sizes keep the `x` form below, where three-to-
# five digits a side make it unambiguous.
_PROSE_RATIO = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*:\s*(\d{1,2}(?:\.\d+)?)(?![\d.])")
_PROSE_SIZE = re.compile(r"(?<!\d)(\d{3,5})\s*[x×]\s*(\d{3,5})(?!\d)", re.I)
_PROSE_HEIGHT = re.compile(r"(?<![\w])(720p|1080p|1440p|2160p|4k|uhd)(?![\w])", re.I)

_HEIGHT_WORDS = {"720p": "720p", "1080p": "1080p", "1440p": "1440p",
                 "2160p": "2160p (4K)", "4k": "2160p (4K)", "uhd": "2160p (4K)"}


def _nearest_ratio(value: float) -> str:
    """The RATIOS key *value* is, or "" when it is not one of them.

    Uses the same tolerance the checker judges by, so inference cannot ask for a
    shape the gate would then call wrong.
    """
    best, best_gap = "", None
    for label, target in RATIOS.items():
        gap = abs(value - target)
        if gap <= RATIO_TOLERANCE * target and (best_gap is None or gap < best_gap):
            best, best_gap = label, gap
    return best


def infer_technical(*texts: str) -> dict:
    """The technical requirements *texts* state outright, as a ``technical`` spec.

    Returns only keys it is sure about, so the result can be merged UNDER an
    explicit dropdown without ever overriding one. Conflicting statements yield
    nothing for that key: two different ratios in one briefing is a question for
    the user, not something to resolve by picking one.
    """
    blob = "\n".join(str(t or "") for t in texts)
    if not blob.strip():
        return {}
    ratios: set = set()
    heights: set = set()

    for a, b in _PROSE_RATIO.findall(blob):
        try:
            num, den = float(a), float(b)
        except ValueError:
            continue
        if den <= 0 or num <= 0:
            continue
        label = _nearest_ratio(num / den)
        if label:
            ratios.add(label)

    for w, h in _PROSE_SIZE.findall(blob):
        try:
            width, height = int(w), int(h)
        except ValueError:
            continue
        if width <= 0 or height <= 0:
            continue
        # A pixel size states a shape and, when it is a standard one, a
        # resolution. Both are read; neither is invented — an odd size that is
        # no named ratio contributes nothing rather than the closest guess.
        label = _nearest_ratio(width / height)
        if label:
            ratios.add(label)
        short = min(width, height)
        for name, value in HEIGHTS.items():
            if short == value:
                heights.add(name)

    for word in _PROSE_HEIGHT.findall(blob):
        name = _HEIGHT_WORDS.get(word.lower())
        if name:
            heights.add(name)

    out: dict = {}
    if len(ratios) == 1:
        out["aspect_ratio"] = ratios.pop()
    elif len(ratios) > 1:
        logger.debug("qa_checks: %d different ratios stated — inferring none", len(ratios))
    if len(heights) == 1:
        out["resolution"] = heights.pop()
    elif len(heights) > 1:
        # Not ambiguous the way two ratios are: "at least" is a floor, and the
        # largest floor satisfies every other one stated.
        out["resolution"] = max(heights, key=lambda n: HEIGHTS[n])
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
    if spec.get("likeness") not in _UNSET:
        lines.append(f"- {spec['likeness']}")
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
