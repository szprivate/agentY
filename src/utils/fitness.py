"""One number for "how good is this render, technically" — for RANKING only.

:mod:`src.utils.qa_checks` answers the questions that have a right answer: is this
16:9, is it 1080p, is it the person in the reference. Those are **gates**, and a
gate is all-or-nothing on purpose.

This is the other half. Given a run of eight variants that all pass the gates,
which one is best? That is not a pass/fail question, it is an ordering, and an
ordering wants a score: normalise every measured property to 0-1, weight them,
add them up.

**Never use this as a gate.** A weighted sum lets a strong feature pay for a weak
one, which is exactly right for "rank these" and exactly wrong for "must be 16:9"
— a beautiful 9:16 render would compensate its way past the requirement. The two
must not be mixed, so the gates stay in :mod:`qa_checks` and never consult this,
and this never returns a verdict.

**The weights are a guess until they are not.** The defaults below are hand-set
and say so; they encode nothing more than "sharp, clean and unclipped beats soft,
grainy and blown". The real ones are meant to be *learned* from what the user
actually picks at a `review` hook — see :mod:`src.utils.preference_log` for the
labels and ``scripts/fit_fitness_weights.py`` for the fit. If that script has run
and beaten the defaults, its weights are in ``config/fitness_weights.json`` and
are loaded here in their place.

**Normalisation is calibrated, not invented.** Every constant below comes from
measuring 260 of this machine's own renders, so a real picture lands somewhere in
the middle of the range instead of pinning at an end where the score stops saying
anything. The percentiles are quoted beside each curve.

A feature that could not be measured is **absent**, not zero: the weights are
renormalised over what is present, so a video (which has no exposure) and a still
are still comparable, and a missing measurement never reads as a defect.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger("agentY.fitness")

WEIGHTS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "fitness_weights.json"

# ── the curves ────────────────────────────────────────────────────────────────
# Measured across 260 real outputs. p2 / p50 / p98 are quoted so the choice of
# range can be checked rather than trusted: each curve puts the median near the
# middle, which is the whole job of a normalisation.

# Variance of the Laplacian is heavy-tailed — a crisp render is not 3x a soft one,
# it is 20x — so both sharpness curves are logarithmic. Linear here would put 95%
# of real output in the bottom fifth of the range.
SHARP_RANGE = (8.0, 220.0)      # global:   p2 9.9   p50 48.1   p98 177.7
FOCUS_RANGE = (20.0, 500.0)     # sharpest: p2 22.9  p50 121.3  p98 404.6
NOISE_RANGE = (0.3, 9.0)        # sigma:    p2 0.40  p50 1.40   p98 6.58
CONTRAST_RANGE = (20.0, 85.0)   # std:      p2 28.1  p50 47.4   p98 74.1
BRIGHTNESS_RANGE = (30.0, 150.0)  # mean:   p2 38.1  p50 70.2   p98 142.2
# Clipping is rare enough to be a defect detector rather than a spread: p98 is
# 0.7% blown and 3.6% crushed, so anything past 5% of the frame scores zero here.
CLIP_RANGE = (0.0, 0.05)

# ── the weights ───────────────────────────────────────────────────────────────
# HAND-SET, and worth no more than that. They say only: sharp, clean and unclipped
# beats soft, grainy and blown; matching the reference matters more than any of
# it; a clip that freezes or goes black is the worst thing a clip can do.
#
# `contrast` and `brightness` are deliberately weighted ZERO. They are style, not
# quality — a dark, low-contrast grade is a choice — so they must not move a
# hand-set score. They are still declared and still measured, because they are
# exactly the kind of preference a fit CAN discover from what someone keeps.
DEFAULT_WEIGHTS = {
    "detail": 0.25,
    "focus": 0.25,
    "cleanliness": 0.20,
    "headroom": 0.15,
    "contrast": 0.0,
    "brightness": 0.0,
    "motion": 0.30,
    "no_black": 0.30,
    "likeness": 0.50,
}

# The order the fitter reads and writes them in. Fixed, because a weight vector
# saved by one version has to mean the same thing when the next one loads it.
FEATURES = ("detail", "focus", "cleanliness", "headroom", "contrast",
            "brightness", "motion", "no_black", "likeness")


def _norm(value, lo: float, hi: float) -> float:
    """*value* on a 0-1 scale between *lo* and *hi*, clamped."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def _log_norm(value, lo: float, hi: float) -> float:
    """As :func:`_norm`, on a log scale — for a metric with a long tail."""
    v = max(0.0, float(value))
    return _norm(math.log10(1.0 + v), math.log10(1.0 + lo), math.log10(1.0 + hi))


def features(facts: dict) -> dict:
    """The measured *facts* as named 0-1 features, higher being better.

    Only what was actually measured appears. A still has no ``motion``; a video
    has no ``headroom`` (no exposure is computed for one); nothing has
    ``likeness`` unless the briefing asked for it. Absent is not zero — see the
    module docstring.
    """
    if not isinstance(facts, dict):
        return {}
    out: dict = {}

    sharp = facts.get("sharpness") or {}
    if sharp.get("score") is not None:
        out["detail"] = _log_norm(sharp["score"], *SHARP_RANGE)
    if sharp.get("sharpest_region") is not None:
        out["focus"] = _log_norm(sharp["sharpest_region"], *FOCUS_RANGE)

    ns = facts.get("noise") or {}
    if ns.get("sigma") is not None:
        out["cleanliness"] = 1.0 - _log_norm(ns["sigma"], *NOISE_RANGE)

    ex = facts.get("exposure") or {}
    if ex:
        clipped = float(ex.get("clipped_white") or 0.0) + float(ex.get("clipped_black") or 0.0)
        out["headroom"] = 1.0 - _norm(clipped, *CLIP_RANGE)
        if ex.get("contrast") is not None:
            out["contrast"] = _norm(ex["contrast"], *CONTRAST_RANGE)
        if ex.get("mean") is not None:
            out["brightness"] = _norm(ex["mean"], *BRIGHTNESS_RANGE)

    sampled = facts.get("frames_sampled")
    if sampled:
        # A stall is counted between consecutive sampled frames, so there are
        # n-1 chances to freeze and only one frame is needed to be black.
        pairs = max(1, int(sampled) - 1)
        out["motion"] = 1.0 - _norm(facts.get("frozen_pairs") or 0, 0, pairs)
        out["no_black"] = 1.0 - _norm(facts.get("black_frames") or 0, 0, int(sampled))

    # Whichever likeness was asked for; they are both already 0-1 and both mean
    # "1.0 is the reference", so one feature covers them.
    for key in ("face_match", "subject_match"):
        m = facts.get(key) or {}
        if m.get("available") and m.get("score") is not None:
            out["likeness"] = max(0.0, min(1.0, float(m["score"])))
            break
    return out


def load_weights() -> dict:
    """The learned weights if a fit has beaten the defaults, else the defaults.

    A file that is missing, unreadable or names nothing recognisable falls back
    silently — a broken weights file must cost the ranking its accuracy, never
    the run.
    """
    try:
        raw = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        learned = {k: float(v) for k, v in (raw.get("weights") or {}).items()
                   if k in DEFAULT_WEIGHTS}
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_WEIGHTS)
    if not learned:
        return dict(DEFAULT_WEIGHTS)
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(learned)
    return merged


def score(facts: dict, weights: dict | None = None) -> dict:
    """``{"score": 0-1, "features": {...}, "weights_source": ...}`` for one output.

    ``{}`` when nothing could be measured — which is not a zero. A file that could
    not be read has no quality, and ranking it last would be a claim we cannot
    make.
    """
    feats = features(facts)
    w = weights if weights is not None else load_weights()
    # Renormalised over the features that are PRESENT. Without this a video would
    # score below every still it is compared with, purely for having no exposure.
    used = {k: float(w.get(k, 0.0)) for k in feats}
    total = sum(abs(v) for v in used.values())
    # Also the "nothing could be measured" exit: no features means no weights to
    # sum. A separate guard for that read better and tested identically, which is
    # a branch pretending to be a decision.
    if total <= 0:
        return {}
    # A FITTED weight may be negative — "darker is preferred" is a real thing to
    # learn — so the score is shifted by the worst reachable total rather than
    # clamped at zero. Clamping would flatten every bad output onto 0.0 and throw
    # away the ordering this exists to produce. With all-positive weights the
    # shift is zero and this is the plain weighted mean.
    floor = sum(v for v in used.values() if v < 0)
    raw = sum(used[k] * v for k, v in feats.items())
    value = (raw - floor) / total
    return {
        "score": round(max(0.0, min(1.0, value)), 3),
        "features": {k: round(v, 3) for k, v in sorted(feats.items())},
        "learned": weights is None and w != DEFAULT_WEIGHTS,
    }


# A review with more outputs than this is not a review anyone is reading one by
# one, and measuring each costs real milliseconds on the turn's critical path.
MAX_RANKED = 24


def score_file(path: str, facts: dict | None = None) -> dict:
    """Measure one file and score it. ``{}`` when it cannot be read.

    *facts* lets a caller that has already measured hand the numbers over instead
    of paying for them twice.
    """
    if facts is None:
        try:
            from src.utils.image_facts import measure
            suffix = Path(str(path)).suffix.lower()
            is_video = suffix in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"}
            facts = measure(str(path), is_video=is_video)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fitness: could not measure %s — %s", path, exc)
            return {}
    return score(facts or {})


def rank_files(paths) -> list:
    """*paths* ordered best-first as ``[{"name", "path", "score", "features"}]``.

    For choosing between the outputs of one run — which is the only comparison
    this score is meant for. Files that could not be measured keep a ``score`` of
    ``None`` and go last: unranked, not worst.
    """
    rows = []
    for p in [str(x) for x in (paths or []) if x][:MAX_RANKED]:
        s = score_file(p)
        rows.append({"name": Path(p).name, "path": p,
                     "score": s.get("score"), "features": s.get("features") or {}})
    measured = [r for r in rows if r["score"] is not None]
    unmeasured = [r for r in rows if r["score"] is None]
    measured.sort(key=lambda r: r["score"], reverse=True)
    return measured + unmeasured


def render_ranking(rows: list) -> str:
    """A ranking as lines for whoever has to describe it — agent or log."""
    if not rows:
        return ""
    out = ["Technical quality ranking (measured, best first). This is a RANKING "
           "aid, not a verdict: a low score is not a failure and the user's own "
           "taste outranks it. Say it only if it helps them choose."]
    for i, r in enumerate(rows, 1):
        if r.get("score") is None:
            out.append(f"  {i}. {r['name']} — could not be measured")
            continue
        weak = sorted((r.get("features") or {}).items(), key=lambda kv: kv[1])[:2]
        why = ", ".join(f"{k} {v:.2f}" for k, v in weak)
        out.append(f"  {i}. {r['name']} — {r['score']:.2f}" + (f" (weakest: {why})" if why else ""))
    return "\n".join(out)


def render_score(s: dict) -> str:
    """The score as one line for the facts block handed to the QA agent."""
    if not s or s.get("score") is None:
        return ""
    parts = ", ".join(f"{k} {v:.2f}" for k, v in (s.get("features") or {}).items())
    return (f"- technical quality score: {s['score']:.2f} of 1.00 "
            f"({parts}) — a RANKING aid only, never a pass or a fail")
