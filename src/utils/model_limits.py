"""Hard input limits the API models enforce, checked before a paid call is made.

Kling 3.0 Omni rejects a prompt over 2,500 characters and more than seven
reference images. ComfyUI raises on both, at execution time, from inside the node
— so the failure arrives as an execution error and goes to the repair specialist,
which is the one agent that cannot help: shortening a prompt without losing what
it was for is rewriting, and picking which of eight references to drop is a
creative choice. Both belong to whoever wrote them. So a violation is detected
deterministically here and handed back to the orchestrator, and the repair turn is
never spent.

The numbers live in ``config/model_limits.json``, read off ComfyUI's own
validators, so adding a model is an edit to that file and not to this one.

Counting images is the harder half: an API prompt carries a LINK on an image
input, not a count, so the number is worked out by walking upstream through the
node classes whose output size is knowable (a loader is one, a batch is the sum of
its parts, a collector is its file list). Anything else makes the count unknown,
and an unknown count never reports a violation — a wrong "too many images" would
cost more than the miss it prevents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

_CONFIG = Path(__file__).parent.parent.parent / "config" / "model_limits.json"
_MAX_HOPS = 24          # upstream walk: deep enough for real graphs, bounded for cycles
_EXCERPT = 90


@dataclass(frozen=True)
class Violation:
    node_id: str
    class_type: str
    field: str
    kind: str           # "text" | "images"
    limit: int
    actual: int
    note: str = ""
    excerpt: str = ""

    def describe(self) -> str:
        where = f"node {self.node_id} ({self.class_type})"
        if self.kind == "text":
            over = self.actual - self.limit
            return (f"{where} `{self.field}` is {self.actual} characters; the model "
                    f"accepts {self.limit}. Cut at least {over}."
                    + (f' Currently starts: "{self.excerpt}"' if self.excerpt else ""))
        return (f"{where} `{self.field}` receives {self.actual} images; the model "
                f"accepts {self.limit}. Drop {self.actual - self.limit}.")


def _config() -> dict:
    """The limits table. Re-read when the file changes so an edit takes effect
    without restarting the host — the whole point of it being data."""
    try:
        stamp = _CONFIG.stat().st_mtime_ns
    except OSError:
        return {}
    cached = getattr(_config, "_cache", None)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        data = json.loads(_CONFIG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        data = {}
    _config._cache = (stamp, data)  # type: ignore[attr-defined]
    return data


def limits_for(class_type: str) -> tuple[dict, dict, str]:
    """``(text_limits, image_limits, note)`` for a node class; empties when unknown."""
    text: dict = {}
    images: dict = {}
    note = ""
    for entry in (_config().get("limits") or []):
        if class_type in (entry.get("nodes") or []):
            text.update(entry.get("text") or {})
            images.update(entry.get("images") or {})
            note = note or str(entry.get("note") or "")
    return text, images, note


def _limit_for_field(limits: dict, field: str) -> int | None:
    """Exact name first, then patterns — so `storyboard_3_prompt` finds
    `storyboard_*_prompt` without every slot being listed."""
    if field in limits:
        return int(limits[field])
    for pattern, value in limits.items():
        if ("*" in pattern or "?" in pattern) and fnmatch(field, pattern):
            return int(value)
    return None


def text_cap(class_type: str, field: str) -> int | None:
    """The character cap this model puts on this input, or None if it has none.

    For whoever is about to WRITE that text: knowing the budget in advance is the
    cheap version of everything else in this module.
    """
    text_limits, _images, _note = limits_for(class_type)
    return _limit_for_field(text_limits, field)


# ── counting the images that arrive on an input ──────────────────────────────

def _sources() -> dict:
    return _config().get("image_sources") or {}


def count_images_from(prompt: dict, node_id: str, _seen: set | None = None,
                      _depth: int = 0) -> int | None:
    """How many images this node emits, or None when that isn't knowable.

    None is a real answer and the common one: most graphs put something in the
    middle that could emit any number. Callers must treat it as "no verdict".
    """
    if _depth > _MAX_HOPS:
        return None
    seen = set(_seen or ())
    if node_id in seen:
        return None                      # a cycle: no verdict, and no hang
    seen.add(node_id)

    node = (prompt or {}).get(str(node_id))
    if not isinstance(node, dict):
        return None
    cls = str(node.get("class_type") or "")
    inputs = node.get("inputs") or {}
    src = _sources()

    if cls in (src.get("one") or []):
        return 1
    if cls in (src.get("collector") or []):
        paths = [ln for ln in str(inputs.get("files") or "").splitlines() if ln.strip()]
        # One file per queue when the collector is stepping through its list.
        if str(inputs.get("load_incrementally", "")).lower() in ("true", "1"):
            return 1 if paths else 0
        return len(paths)
    if cls in (src.get("passthrough") or []):
        for value in inputs.values():
            if isinstance(value, list) and value:
                return count_images_from(prompt, str(value[0]), seen, _depth + 1)
        return None
    if cls in (src.get("sum") or []):
        total = 0
        for value in inputs.values():
            if not (isinstance(value, list) and value):
                continue
            part = count_images_from(prompt, str(value[0]), seen, _depth + 1)
            if part is None:
                return None              # one unknown part makes the sum unknown
            total += part
        return total or None
    return None


def count_images_into(prompt: dict, node_id: str, input_name: str) -> int | None:
    """How many images arrive on *input_name* of *node_id*, or None if unknowable."""
    node = (prompt or {}).get(str(node_id))
    if not isinstance(node, dict):
        return None
    value = (node.get("inputs") or {}).get(input_name)
    if not (isinstance(value, list) and value):
        return None                      # not wired, or a widget: nothing to count
    return count_images_from(prompt, str(value[0]))


# ── the check ────────────────────────────────────────────────────────────────

def check_workflow(prompt: dict) -> list[Violation]:
    """Every hard limit this workflow breaks, worst overrun first."""
    out: list[Violation] = []
    if not isinstance(prompt, dict):
        return out
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        text_limits, image_limits, note = limits_for(cls)
        if not text_limits and not image_limits:
            continue
        inputs = node.get("inputs") or {}

        for field, value in inputs.items():
            if not isinstance(value, str):
                continue                 # a link, or a number: not text we set
            cap = _limit_for_field(text_limits, field)
            if cap is None or len(value) <= cap:
                continue
            out.append(Violation(str(node_id), cls, field, "text", cap, len(value),
                                 note, value[:_EXCERPT].replace("\n", " ")))

        for field, cap in image_limits.items():
            n = count_images_into(prompt, str(node_id), field)
            if n is not None and n > int(cap):
                out.append(Violation(str(node_id), cls, field, "images", int(cap), n, note))

    return sorted(out, key=lambda v: v.actual - v.limit, reverse=True)


def check_value(prompt: dict, node_id: str, field: str, value) -> Violation | None:
    """One value about to be written into one input — checked before it is written.

    This is the version that matters on the canvas path, where the agent writes a
    value and a tool accepts it. Told at the moment of writing, the agent that just
    wrote it can rewrite it inside the same turn; told afterwards, the run is
    already queued and the only thing left to do is apologise to the user.
    """
    if not isinstance(value, str):
        return None
    node = (prompt or {}).get(str(node_id))
    cls = str(node.get("class_type") or "") if isinstance(node, dict) else ""
    if not cls:
        return None
    text_limits, _images, note = limits_for(cls)
    cap = _limit_for_field(text_limits, str(field))
    if cap is None or len(value) <= cap:
        return None
    return Violation(str(node_id), cls, str(field), "text", cap, len(value),
                     note, value[:_EXCERPT].replace("\n", " "))


def check_workflow_file(path: str) -> list[Violation]:
    try:
        return check_workflow(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return []


# ComfyUI's own wording when a node raises at run time. Recognising it lets the
# repair path bail out even for a model that isn't in the table yet — the message
# is the evidence, and no repair the specialist can make will change it.
_RUNTIME_PATTERNS = (
    re.compile(r"cannot be longer than\s+(\d+)\s+characters.*?was\s+(\d+)", re.I | re.S),
    re.compile(r"prompt is too long:\s*(\d+)\s*characters", re.I),
    re.compile(r"maximum number of supported images is\s+(\d+)", re.I),
    re.compile(r"maximum of\s+(\d+)\s+input images", re.I),
    re.compile(r"supports only\s+(\d+)\s+input image", re.I),
    re.compile(r"(?:no more than|at most|maximum)\s+(\d+)\s+(?:reference )?images", re.I),
)


def runtime_limit_error(exec_error: dict | None) -> str:
    """The model's own complaint about an input limit, or '' if it said something else.

    ComfyUI raises these from inside the node, so they arrive as an ordinary
    execution error and would otherwise be handed to the repair specialist to
    "fix" — which it cannot, however many turns it spends.
    """
    if not isinstance(exec_error, dict):
        return ""
    det = exec_error.get("details") or {}
    text = " ".join(str(x) for x in (
        det.get("exception_message", ""), exec_error.get("error", ""),
        det.get("exception_type", ""),
    ) if x)
    for pattern in _RUNTIME_PATTERNS:
        if pattern.search(text):
            return " ".join(text.split())[:300]
    return ""


def summary(violations: list, runtime_message: str = "") -> str:
    """One line for a log or a batch failure report, where nobody can act on it.

    The batch executor heals members mid-run and prints whatever ``error`` comes
    back; without this it prints "still invalid", which is true and useless.
    """
    if violations:
        parts = [f"{v.field} {v.actual}"
                 f"{' chars' if v.kind == 'text' else ' images'} > {v.limit}"
                 for v in violations]
        head = violations[0]
        return (f"{head.class_type} refuses this input (hard model limit): "
                + "; ".join(parts) + " — the prompt/inputs need rewriting, not repair.")
    return (f"the model refused this input (hard limit): {runtime_message}"
            if runtime_message else "the model refused this input (hard limit).")


def canvas_refusal(violations: list, attempt: int = 1) -> dict:
    """The tool result for a canvas write that would not survive the run.

    Returned instead of accepting the value, so the agent that wrote it gets the
    number while it is still holding the turn. It has everything it needs to fix
    this itself — it wrote the text — so the one thing this must not do is end up
    in front of the user as an apology.
    """
    worst = violations[0]
    if worst.kind == "text":
        what = (f"You wrote {worst.actual} characters into `{worst.field}` on node "
                f"{worst.node_id} ({worst.class_type}); it accepts {worst.limit}. "
                f"Cut at least {worst.actual - worst.limit} and call this tool again.")
        how = ("Rewrite rather than truncate — a prompt cut mid-sentence reads as a "
               "different instruction. Drop restatement and stacked synonyms first, "
               "then the least load-bearing detail. If the content genuinely will not "
               "fit, SPLIT it across several runs of the node rather than sending one "
               "over-long value.")
    else:
        what = (f"You wired {worst.actual} images into `{worst.field}` on node "
                f"{worst.node_id} ({worst.class_type}); it accepts {worst.limit}. "
                f"Drop {worst.actual - worst.limit} and call this tool again.")
        how = ("Keep the references the result depends on. Say in your reply which you "
               "dropped, so the user can disagree.")
    if attempt >= 3:
        # Repeating the same sentence a third time is not advice. Something about
        # the content does not fit, and the way out is structural.
        how = (f"This is attempt {attempt} on the same input, so stop trimming: the "
               "content does not fit and will not start to. Either SPLIT it across "
               "several runs of the node (one value per run, each inside the limit) "
               "or tell the user what you would have to cut and let them choose. Do "
               "not send the same shape of value a fourth time.")
    return {
        "error": "rejected by the model's hard input limit — nothing was placed or queued",
        "what_to_fix": what,
        "how": how,
        "attempt": attempt,
        "violations": [{"node_id": v.node_id, "class_type": v.class_type,
                        "field": v.field, "kind": v.kind, "limit": v.limit,
                        "actual": v.actual} for v in violations],
        "do_not": "Do not report this to the user as a failure and do not stop the "
                  "turn: it is yours to fix, now, by writing a shorter value.",
    }


def guidance(violations: list, runtime_message: str = "") -> str:
    """What the orchestrator should do about it, in the terms it can act in."""
    lines = ["The model rejects this input — it is a hard API limit, not a workflow "
             "defect, so the repair specialist cannot fix it."]
    if violations:
        lines += [f"  - {v.describe()}" for v in violations]
    elif runtime_message:
        lines.append(f"  - the model reported: {runtime_message}")
    lines += [
        "",
        "Fix it yourself, then continue:",
        "  - TOO LONG: rewrite the prompt under the limit, keeping the subject, the "
        "action and the look. Cut restatement and stacked synonyms first, then the "
        "least load-bearing detail. Do NOT truncate mid-sentence — a cut-off prompt "
        "reads as a different instruction.",
        "  - TOO MANY IMAGES: keep the ones the result depends on, drop the rest, and "
        "say in your reply which you dropped so the user can disagree.",
        "  - Apply the change with update_workflow(workflow_path, patches=[…]), then "
        "call signal_workflow_ready(workflow_path). Do not re-run prepare_workflow: "
        "the workflow is otherwise assembled and valid.",
    ]
    return "\n".join(lines)
