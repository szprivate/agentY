"""Which knob in the graph governs a requirement — and what to turn it to.

QA can say an output is the wrong shape with certainty. The retry that follows
could not do anything about it: it rerolls the seed and rewrites the positive
prompt, and neither of those has ever changed an image's dimensions. So a 1:1
render against a *16:9* briefing burned a real, paid generation and came back
failing with the identical number.

The fix is not cleverer prose. It is to find the parameter that decides the thing
that failed and set it correctly — which is possible because the two vocabularies
already coincide. ``KlingTextToVideoNode`` declares ``aspect_ratio`` as a combo
whose options are ``["16:9", "9:16", "1:1"]``: the same strings the briefing node
offers. ``EmptyLatentImage`` declares ``width`` and ``height``. Of the 1,735 node
classes on this install, 275 declare something in this family.

**Which node.** A graph can name a size in several places — a latent, the
generator, a resize on the way to the saver — and only one of them is the shape
the picture is *made* at. Rescaling afterwards would satisfy the measurement and
misrepresent the render, so the walk goes upstream from the output and takes the
**furthest** governing node it finds: the size source feeding the generator,
never a downstream resize. Where the generator carries the parameter itself (the
API nodes), it is the only candidate and wins by default.

**What it will not do.** Sharpness, grain, clipping, likeness, black frames — no
parameter governs any of them; they are properties of the picture, not settings.
This says so rather than guessing, and saying so is most of the value: a retry
that cannot work should not be paid for.

Used twice, for the same reason at two different moments:

* **before** a run, so a requirement the graph contradicts is fixed (or reported)
  without spending anything — see :func:`plan_fixes`;
* **after** a failed QA check, so the retry changes the thing that was wrong
  instead of the two things that were not.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("agentY.qa_repair")

# The briefing controls that a graph parameter can actually decide. Everything
# else in a briefing is a property of the picture and belongs to the judge.
GOVERNED = ("aspect_ratio", "resolution")

# Widget names that carry a picture's shape, in the order we would rather use
# them: a named ratio is exact, a size preset is a menu, width/height is
# arithmetic. Matched on the whole name, so `batch_size` and `size_preset` do not
# collide by accident.
RATIO_NAMES = ("aspect_ratio", "ratio", "aspect")
SIZE_NAMES = ("size", "size_preset", "image_size", "resolution", "dimensions")
DIM_NAMES = ("width", "height")

# Computed sizes land on multiples of this. 8 is the hard requirement — a latent
# side that is not a multiple of 8 fails — while 64 is a preference some pipelines
# have. Snapping to 64 would rewrite 1080 to 1088: a size the user chose, that
# works, and that is exactly 1080p. Imposing a preference on someone else's render
# is not this module's job; not breaking it is.
SNAP = 8
MIN_SIDE = 256

_RATIO_TEXT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[:x×]\s*(\d+(?:\.\d+)?)\s*$")
_SIZE_TEXT = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")


def _combo_options(spec) -> list:
    """The options of a combo input, whichever shape the node declared it in."""
    if isinstance(spec, list) and spec:
        head = spec[0]
        if isinstance(head, list):
            return [str(o) for o in head]
        if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [str(o) for o in (spec[1].get("options") or [])]
    return []


def _inputs(schema: dict) -> dict:
    """``{name: spec}`` over required and optional inputs together."""
    io = schema.get("input") or {}
    out = {}
    out.update(io.get("required") or {})
    out.update(io.get("optional") or {})
    return out


def _is_int(spec) -> bool:
    return isinstance(spec, list) and spec and spec[0] == "INT"


def ratio_of(text: str) -> float | None:
    """The numeric ratio a label means: ``"16:9"`` → 1.778, ``"1024x576"`` → 1.778."""
    m = _RATIO_TEXT.match(str(text or ""))
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        return (w / h) if h else None
    m = _SIZE_TEXT.search(str(text or ""))
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        return (w / h) if h else None
    return None


def short_side_of(text: str) -> int | None:
    """The short side a size label implies, when it names actual pixels."""
    m = _SIZE_TEXT.search(str(text or ""))
    if m:
        return min(int(m.group(1)), int(m.group(2)))
    return None


# ── finding the node ────────────────────────────────────────────────────────

def _ancestors(prompt: dict, start: str) -> dict:
    """``{node_id: depth}`` for everything feeding *start*, *start* included.

    Depth is how many links away, so "furthest upstream" is simply the largest.
    """
    out = {str(start): 0}
    stack = [(str(start), 0)]
    while stack:
        nid, d = stack.pop()
        node = (prompt or {}).get(nid)
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2:
                up = str(value[0])
                if up not in out or out[up] < d + 1:
                    out[up] = d + 1
                    stack.append((up, d + 1))
    return out


def _terminals(prompt: dict) -> list:
    """The output nodes a run ends at — where the walk upstream begins."""
    from src.utils.canvas_hooks import is_terminal
    return [nid for nid, node in (prompt or {}).items()
            if isinstance(node, dict) and is_terminal(node.get("class_type"))]


def governing_params(prompt: dict, schema_of=None) -> list:
    """Every parameter in *prompt* that decides an output's shape.

    ``[{node_id, class_type, kind, param(s), value(s), depth}]``, furthest
    upstream first — which is the order of preference, because the size a picture
    is *generated* at sits above every resize on the way to the saver.
    """
    if schema_of is None:
        from src.utils.preflight import _schema as schema_of
    if not isinstance(prompt, dict):
        return []
    depth: dict = {}
    for term in _terminals(prompt):
        for nid, d in _ancestors(prompt, term).items():
            depth[nid] = max(depth.get(nid, 0), d)

    found = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        try:
            declared = _inputs(schema_of(cls))
        except Exception as exc:  # noqa: BLE001 — a node we cannot ask about is skipped
            logger.debug("qa_repair: no schema for %s — %s", cls, exc)
            continue
        if not declared:
            continue
        values = node.get("inputs") or {}
        row = {"node_id": str(nid), "class_type": cls, "depth": depth.get(str(nid), 0)}

        name = next((n for n in RATIO_NAMES if n in declared), "")
        if name and not isinstance(values.get(name), list):
            opts = _combo_options(declared[name])
            if opts:
                found.append({**row, "kind": "ratio", "param": name,
                              "options": opts, "value": values.get(name)})
                continue

        name = next((n for n in SIZE_NAMES if n in declared), "")
        if name and not isinstance(values.get(name), list):
            opts = _combo_options(declared[name])
            if opts:
                found.append({**row, "kind": "size", "param": name,
                              "options": opts, "value": values.get(name)})
                continue

        if all(n in declared and _is_int(declared[n]) for n in DIM_NAMES) and \
                not any(isinstance(values.get(n), list) for n in DIM_NAMES):
            found.append({**row, "kind": "dims", "params": list(DIM_NAMES),
                          "value": {n: values.get(n) for n in DIM_NAMES}})

    # Furthest upstream wins: that is the size the picture is made at, and a
    # resize nearer the saver would satisfy the ruler while misreporting the
    # render. Node id breaks a tie so the same graph always picks the same node.
    found.sort(key=lambda r: (-r["depth"], str(r["node_id"])))
    return found


# ── choosing the value ──────────────────────────────────────────────────────

def _best_option(options: list, want_ratio: float | None, want_short: int | None,
                 tolerance: float = 0.02):
    """The option that satisfies the requirement, or None if none does."""
    scored = []
    for opt in options:
        r = ratio_of(opt)
        s = short_side_of(opt)
        if want_ratio is not None:
            if r is None or abs(r - want_ratio) > tolerance * want_ratio:
                continue
        if want_short is not None and s is not None and s < want_short:
            continue
        # Prefer the smallest option that qualifies: it is the cheapest render
        # that meets what was asked, and asking for more than that is our choice
        # to make on someone else's bill.
        scored.append(((s or 0), opt))
    if not scored:
        return None
    scored.sort()
    return scored[0][1]


def _snap(value: float) -> int:
    return max(MIN_SIDE, int(round(value / SNAP)) * SNAP)


def _dims_for(current: dict, want_ratio: float | None, want_short: int | None):
    """``(width, height)`` meeting the requirement, or None when already right.

    Pixel count is preserved where it can be: the user chose a render size and a
    reshape should not quietly double what it costs.
    """
    try:
        w = int(current.get("width") or 0)
        h = int(current.get("height") or 0)
    except Exception:  # noqa: BLE001
        return None
    if w <= 0 or h <= 0:
        return None
    # Already right? Then there is nothing to change, and saying so matters: an
    # edit that moves 1920x1080 to 1920x1088 reports a fix for a graph that was
    # correct, which teaches people to distrust the fixes that are real.
    from src.utils.qa_checks import RATIO_TOLERANCE
    ratio_ok = want_ratio is None or abs((w / h) - want_ratio) <= RATIO_TOLERANCE * want_ratio
    short_ok = want_short is None or min(w, h) >= want_short
    if ratio_ok and short_ok:
        return None
    if want_ratio:
        area = float(w * h)
        nh = (area / want_ratio) ** 0.5
        nw = nh * want_ratio
    else:
        nw, nh = float(w), float(h)
    if want_short:
        short = min(nw, nh)
        if short < want_short:
            scale = want_short / short
            nw, nh = nw * scale, nh * scale
    out = (_snap(nw), _snap(nh))
    return None if out == (w, h) else out


def evaluate_shape(prompt: dict, want_ratio: float | None, want_short: int | None,
                   why: str = "", schema_of=None) -> tuple:
    """``(status, fix)`` for the shape a graph will produce.

    Four answers, because they mean four different things to whoever reads them:

    * ``"fix"`` — an edit to make;
    * ``"satisfied"`` — a parameter governs the shape and already gives it;
    * ``"ungoverned"`` — nothing in the graph decides it at all;
    * ``"unreachable"`` — something decides it, and cannot reach what was asked.

    The last two look identical from here (no edit planned) and are opposite
    advice. `OpenAIDalle2` only offers squares, so a 16:9 briefing is not a
    missing parameter to go and find — it is a model that cannot make that
    picture, and telling someone to look for the knob wastes their afternoon.
    "Satisfied" is kept apart for the same reason in reverse: reporting a problem
    on a graph that will pass teaches people to ignore the ones that are real.

    Ratio and resolution arrive **together** because they are one decision about
    one thing. Planned separately they overwrite each other: 1024x1024 becomes
    1344x768 for *16:9*, then 1088x1088 for *1080p*, and the second edit throws
    away the first.
    """
    if want_ratio is None and want_short is None:
        return ("ungoverned", None)

    tried: list = []
    for cand in governing_params(prompt, schema_of):
        tried.append(cand)
        if cand["kind"] in ("ratio", "size"):
            # A menu that names ratios cannot also be asked for a pixel count, and
            # one that names sizes can be asked for both. Hold the ratio it
            # already has when only the resolution was asked for, so satisfying
            # one requirement never silently reshapes the picture.
            here_ratio = (want_ratio if want_ratio is not None
                          else ratio_of(str(cand.get("value") or "")))
            pick = _best_option(cand["options"], here_ratio, want_short)
            if pick is None:
                continue                      # this menu has nothing that fits
            if str(pick) == str(cand.get("value")):
                return ("satisfied", None)
            return ("fix", {"node_id": cand["node_id"], "class_type": cand["class_type"],
                            "param": cand["param"], "from": cand.get("value"),
                            "to": pick, "why": why})
        if cand["kind"] == "dims":
            dims = _dims_for(cand["value"], want_ratio, want_short)
            if not dims:
                return ("satisfied", None)    # the arithmetic already lands there
            return ("fix", {"node_id": cand["node_id"], "class_type": cand["class_type"],
                            "params": ["width", "height"], "from": dict(cand["value"]),
                            "to": {"width": dims[0], "height": dims[1]}, "why": why})
    # Candidates existed but none could be made to fit: the menus on offer do not
    # contain the shape asked for, which is the model's answer, not ours.
    if tried:
        return ("unreachable", {"node_id": tried[0]["node_id"],
                                "class_type": tried[0]["class_type"],
                                "param": tried[0].get("param") or "width/height",
                                "options": tried[0].get("options") or []})
    return ("ungoverned", None)


def _wanted(technical: dict) -> tuple:
    """``(want_ratio, want_short, why)`` from a briefing's stated controls."""
    from src.utils.qa_checks import HEIGHTS, RATIOS, _UNSET

    tech = technical if isinstance(technical, dict) else {}
    ratio_label = tech.get("aspect_ratio")
    short_label = tech.get("resolution")
    want_ratio = RATIOS.get(str(ratio_label)) if ratio_label not in _UNSET else None
    want_short = HEIGHTS.get(str(short_label)) if short_label not in _UNSET else None
    why = " and ".join(
        ([f"aspect ratio {ratio_label}"] if want_ratio is not None else [])
        + ([f"at least {short_label}"] if want_short is not None else []))
    return want_ratio, want_short, why


def evaluate_control(prompt: dict, control: str, want: str, schema_of=None) -> tuple:
    """``(status, fix)`` for ONE requirement. See :func:`evaluate_shape`.

    A control outside :data:`GOVERNED` needs no guard here: `_wanted` reads only
    the two keys it knows, so anything else yields no constraint and falls out of
    `evaluate_shape` as ungoverned on its own.
    """
    want_ratio, want_short, why = _wanted({control: want})
    return evaluate_shape(prompt, want_ratio, want_short, why, schema_of)


def plan_fix(prompt: dict, control: str, want: str, schema_of=None) -> dict | None:
    """The edit that makes *prompt* satisfy one requirement, or ``None``."""
    status, fix = evaluate_control(prompt, control, want, schema_of)
    return fix if status == "fix" else None


def apply_fix(prompt: dict, fix: dict) -> bool:
    """Write one planned fix into *prompt* in place. True when it landed."""
    node = (prompt or {}).get(str((fix or {}).get("node_id") or ""))
    if not isinstance(node, dict):
        return False
    inputs = node.setdefault("inputs", {})
    if fix.get("param"):
        inputs[fix["param"]] = fix["to"]
        return True
    for name, value in (fix.get("to") or {}).items():
        inputs[name] = value
    return bool(fix.get("to"))


def describe_fix(fix: dict) -> str:
    """One line naming what was changed and on whose authority.

    Deliberately ASCII. This string reaches the chat panel, the executor's log and
    a `print` to a Windows console, and that last one is cp1252: an arrow here
    raised UnicodeEncodeError *inside the retry*, turning a fix into a crash.
    """
    if not fix:
        return ""
    where = f"node {fix['node_id']} ({fix['class_type']})"
    if fix.get("param"):
        return (f"{where}: {fix['param']} {fix.get('from')!r} -> {fix['to']!r} "
                f"(your briefing asks for {fix['why']})")
    a, b = fix.get("from") or {}, fix.get("to") or {}
    return (f"{where}: {a.get('width')}x{a.get('height')} -> "
            f"{b.get('width')}x{b.get('height')} (your briefing asks for {fix['why']})")


def plan_fixes(prompt: dict, technical: dict, schema_of=None) -> tuple:
    """``(fixes, problems)`` for the shape requirements a briefing states.

    Each problem is ``{control, status, why}`` — ``why`` being a sentence worth
    showing, since "nothing sets this" and "the model cannot make that shape" send
    a reader to two different places.
    """
    from src.utils.qa_checks import _UNSET

    tech = technical if isinstance(technical, dict) else {}
    want_ratio, want_short, why = _wanted(tech)
    asked = [c for c in GOVERNED if tech.get(c) not in _UNSET]
    if not asked:
        # Not for the answer's sake — `evaluate_shape` would reach the same one —
        # but for the cost of getting there: it walks the graph asking ComfyUI for
        # a schema per class, and a briefing about sharpness has not asked a
        # question that needs any of it.
        return [], []
    try:
        status, payload = evaluate_shape(prompt, want_ratio, want_short, why, schema_of)
    except Exception as exc:  # noqa: BLE001 — never worth a turn
        logger.debug("qa_repair: could not plan the shape — %s", exc)
        return [], []
    if status == "fix":
        return [payload], []
    if status == "satisfied":
        return [], []
    if status == "unreachable":
        where = f"node {payload['node_id']} ({payload['class_type']}) sets it via " \
                f"`{payload['param']}`"
        offers = payload.get("options") or []
        detail = (f", and offers only {', '.join(offers)}" if offers
                  else ", and cannot reach it")
        reason = f"{where}{detail}"
    else:
        reason = "nothing in this graph sets it"
    return [], [{"control": c, "status": status, "why": reason} for c in asked]
