"""The refine loop: run the graph the user has open, judge it, change one value, run again.

This is the *panel-mode* counterpart to the ``iterate`` hook. The hook version
(``Pipeline.iterate_step``) is a conversation — one generation per turn, the user
looking at each result and saying what to change next. This one is a **closed
loop**: the user states a condition once ("until the woman's position matches the
original frame"), and the agent runs the graph, judges the output against that
condition, rewrites the value it was told to vary, and runs again — until the
condition is met or the budget is spent.

Three things had to be true for that to be safe to build, and each one is a
decision recorded here rather than in the tool that calls it:

**It runs the graph the user has open, not a copy of it.** Panel mode's standing
rule is that agentY edits the canvas and the *user* queues it (see
``orchestrator/canvas_nodes.md``). A loop is the one case where that cannot hold —
the whole point is to look at the output — so the loop is entered only when the
user asks for one, and what it queues is a patched deep copy of their own graph.
No template is picked, nothing is assembled, no second workflow appears.

**It changes exactly one value, and says which.** A loop that quietly rewrote
several inputs would make its own result unreadable: nobody could say which change
earned the pass. :func:`choose_target` picks that value or refuses with the list of
candidates, and it will not auto-select a NEGATIVE prompt — refining "what I don't
want" against a goal phrased as "what I do want" inverts the whole loop.

**A judge that cannot be read must not be able to declare victory.**
``qa.check_output`` passes on doubt, deliberately: an unreachable judge must never
condemn the user's work. In a loop that same rule is inverted — a pass on doubt
would stop the loop and report a success nobody verified. :func:`verdict_of` keeps
the two apart and gives the unreadable judge a status of its own.

The mechanics that need no LLM live here, so they can be tested without ComfyUI,
without a model and without the pipeline. The driver — which owns the executor,
the canvas patch and the budget — is ``refine_canvas_until`` in ``pipeline.py``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# How many generations one loop may spend when the settings file says nothing.
# Each one is a real render and, on a hosted model, real money — so the budget is
# a ceiling the agent may go under and never over.
DEFAULT_MAX_RUNS = 4

# Widget names that are never a prompt, whatever they happen to hold. A combo of
# checkpoint names is a string like any other; letting the loop rewrite one would
# swap the user's model out from under them mid-loop.
_NEVER_TEXT = {
    "ckpt_name", "vae_name", "lora_name", "control_net_name", "clip_name",
    "clip_name1", "clip_name2", "clip_name3", "unet_name", "style_model_name",
    "gligen_name", "upscale_model_name", "model_name", "model", "sampler_name",
    "scheduler", "filename_prefix", "image", "images", "video", "audio", "mask",
    "preset", "device", "dtype", "weight_dtype", "precision", "format", "type",
    "mode", "method", "resampling", "interpolation", "crop", "upscale_method",
    "add_noise", "return_with_leftover_noise", "api_key", "auth_token",
    "aspect_ratio", "resolution", "size", "quality", "background", "duration",
}

# Widget names that ARE a prompt slot even when empty, and which no length or
# whitespace heuristic should have to argue for.
_PROMPT_NAMES = {
    "text", "prompt", "positive", "negative", "string", "value", "text_g",
    "text_l", "text_positive", "text_negative", "positive_prompt",
    "negative_prompt", "caption", "description", "instruction",
}

# A value ending in one of these names a file on disk, not something to write.
_ASSET_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
                   ".onnx", ".yaml", ".yml", ".json", ".txt"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
_MEDIA_SUFFIXES = _IMAGE_SUFFIXES | {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# A free-text value has to look like prose before an unnamed widget counts as a
# prompt: "euler_ancestral" and "fp8_e4m3fn" are strings too.
_MIN_PROSE = 12

_ANNOTATED = re.compile(r"\s*\[(?:input|output|temp)\]\s*$")


# -- settings -------------------------------------------------------------------

def max_runs_cap() -> int:
    """The most generations one loop may spend, from ``[refine] max_runs``.

    Read fresh each call rather than cached: this is the knob a user reaches for
    right after a loop cost them more than they meant to spend, and it should take
    effect on the next loop rather than the next restart.
    """
    try:
        from src.utils.settings import load_settings
        raw = (load_settings().get("refine") or {}).get("max_runs", DEFAULT_MAX_RUNS)
        return max(1, int(raw))
    except Exception:  # noqa: BLE001 — a bad settings file must not break a turn
        return DEFAULT_MAX_RUNS


def clamp_runs(asked) -> tuple[int, int]:
    """``(runs, cap)`` — what the loop will actually spend, and the ceiling.

    Returning both is what lets the tool say *why* it is doing four when it was
    asked for ten, instead of silently doing four.
    """
    cap = max_runs_cap()
    try:
        want = int(asked)
    except (TypeError, ValueError):
        want = 0
    if want < 1:
        want = cap
    return min(want, cap), cap


# -- which value the loop varies ------------------------------------------------

def _title(node: dict) -> str:
    meta = node.get("_meta") if isinstance(node, dict) else None
    return str(meta.get("title") or "") if isinstance(meta, dict) else ""


def _negative_node_ids(graph: dict) -> set[str]:
    """Nodes whose output feeds an input called something-negative.

    Read off the wiring rather than off the node's own name, because a
    CLIPTextEncode has no idea which of the two it is — only the sampler it feeds
    knows. A node feeding both ends up in here as well, which is the right answer:
    a value that is positive and negative at once is not one to pick unasked.
    """
    out: set[str] = set()
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if (isinstance(value, list) and len(value) == 2
                    and "negative" in str(name).lower()):
                out.add(str(value[0]))
    return out


def _is_texty(name: str, value) -> bool:
    """Whether widget *name* holding *value* is something a prompt could live in."""
    if not isinstance(value, str):
        return False
    key = str(name).lower()
    if key in _NEVER_TEXT:
        return False
    text = value.strip()
    if not text:
        return key in _PROMPT_NAMES     # an empty prompt box is still a prompt box
    suffix = Path(_ANNOTATED.sub("", text)).suffix.lower()
    if suffix in _ASSET_SUFFIXES or suffix in _MEDIA_SUFFIXES:
        return False
    if key in _PROMPT_NAMES:
        return True
    return " " in text and len(text) >= _MIN_PROSE


def text_targets(graph: dict | None) -> list[dict]:
    """Every widget on the graph that could carry a prompt, likeliest first.

    Each entry carries ``node_id``, ``param``, ``class_type``, ``title``, ``value``
    and ``role`` (``positive`` / ``negative``). Order is by role, then by how much
    text is in it — the longest positive prompt is nearly always the one the user
    means, and putting it first is what makes a "several candidates" refusal
    readable instead of a wall.
    """
    negatives = _negative_node_ids(graph or {})
    found: list[dict] = []
    for node_id, node in (graph or {}).items():
        if not isinstance(node, dict):
            continue
        title = _title(node)
        for name, value in (node.get("inputs") or {}).items():
            if not _is_texty(name, value):
                continue
            key = str(name).lower()
            negative = (str(node_id) in negatives
                        or "negative" in key
                        or "negative" in title.lower())
            found.append({
                "node_id": str(node_id),
                "param": str(name),
                "class_type": str(node.get("class_type") or ""),
                "title": title,
                "value": value,
                "role": "negative" if negative else "positive",
            })
    found.sort(key=lambda t: (t["role"] != "positive", -len(str(t["value"]))))
    return found


def describe_targets(targets: list) -> list[str]:
    """The candidates as lines a person can act on, values kept to one line."""
    out = []
    for t in targets or []:
        text = " ".join(str(t.get("value") or "").split())
        if len(text) > 70:
            text = text[:70].rstrip() + "…"
        name = t.get("title") or t.get("class_type") or "node"
        role = "" if t.get("role") == "positive" else f" [{t.get('role')}]"
        out.append(f"{t.get('node_id')}.{t.get('param')} — {name}{role}: \"{text}\"")
    return out


def choose_target(graph: dict | None, node_id: str = "",
                  param: str = "") -> tuple[dict | None, dict | None]:
    """``(target, error)`` — the one value this loop will vary.

    Refuses rather than guesses whenever more than one reading is possible, and
    every refusal carries the candidates with it, so the answer to one is a second
    call with a name in it and never a hunt across the canvas.
    """
    graph = graph if isinstance(graph, dict) else {}
    targets = text_targets(graph)
    if node_id:
        nid = str(node_id)
        if nid not in graph:
            return None, {"error": f"there is no node {nid} on the open canvas.",
                          "text_widgets_on_the_canvas": describe_targets(targets)}
        on_node = [t for t in targets if t["node_id"] == nid]
        if param:
            hit = next((t for t in on_node if t["param"] == str(param)), None)
            if hit:
                return hit, None
            raw = (graph[nid].get("inputs") or {}).get(str(param))
            if isinstance(raw, list):
                return None, {"error": f"node {nid}'s `{param}` is a WIRED input, not a "
                                       "value — the loop can only change something you "
                                       "could type into."}
            if raw is not None:
                return None, {"error": f"node {nid}'s `{param}` is not text the loop can "
                                       f"rewrite (it holds {json.dumps(raw)[:60]}).",
                              "text_widgets_on_this_node": [t["param"] for t in on_node]}
            return None, {"error": f"node {nid} has no widget called `{param}`.",
                          "text_widgets_on_this_node": [t["param"] for t in on_node]}
        if len(on_node) == 1:
            return on_node[0], None
        if not on_node:
            return None, {"error": f"node {nid} carries no text widget to vary.",
                          "text_widgets_on_the_canvas": describe_targets(targets)}
        return None, {"error": f"node {nid} has several text widgets — say which one in "
                               "`param`.",
                      "choices": describe_targets(on_node)}

    positives = [t for t in targets if t["role"] == "positive"]
    if len(positives) == 1:
        return positives[0], None
    if not positives:
        return None, {
            "error": "nothing on the open canvas looks like a prompt to vary — name the "
                     "node and the widget yourself with `node_id` and `param`.",
            "text_widgets_on_the_canvas": (describe_targets(targets)
                                           or "(no text widget found at all)")}
    return None, {"error": "several nodes carry a prompt — say which one to vary in "
                           "`node_id` (and `param` too, if that node has more than one).",
                  "candidates": describe_targets(positives)}


# -- what the loop compares against ---------------------------------------------

def graph_reference_images(graph: dict | None, resolver=None, limit: int = 4) -> list[str]:
    """The images the graph itself loads, as absolute paths.

    "Until it matches the original frame" names something the user never has to
    hand over: the frame is already sitting in a loader on their canvas. Resolving
    those is what lets the condition be written the way they would say it out loud.
    Explicit ``references`` always win over this — it is a fallback, not an opinion.
    """
    out: list[str] = []
    for node in (graph or {}).values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip().strip('"')
            # ComfyUI annotates a loader's value as "name.png [input]".
            if Path(_ANNOTATED.sub("", text)).suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            resolved = resolver(text, "image") if resolver else text
            if resolved and str(resolved) not in out:
                out.append(str(resolved))
            if len(out) >= limit:
                return out
    return out


def verdict_of(result) -> tuple[str, str, list[str]]:
    """``(status, summary, failures)`` for one judged output.

    ``qa.check_output`` answers a different question from the one a loop asks. It
    passes on doubt, because an unreachable judge must not condemn the user's
    work — but a loop reads a pass as "stop, you are done", so that same doubt
    would end the loop on a verdict nobody wrote. ``unjudged`` keeps the two apart,
    and is why a broken judge stops the loop instead of finishing it in success.
    """
    error = str(getattr(result, "error", "") or "")
    summary = str(getattr(result, "summary", "") or "")
    if error:
        return "unjudged", error, []
    if getattr(result, "passed", False):
        return "matched", summary, []
    failures = list(result.failed_criteria()) if hasattr(result, "failed_criteria") else []
    if not failures and summary:
        failures = [summary]
    return "missed", summary, failures


# -- rewriting the value --------------------------------------------------------

def load_refine_prompts() -> dict[str, str]:
    """The reviser's prompt file, parsed into its ``## <name>`` sections.

    Same shape as the QA prompts, and for the same reason: what the reviser is told
    is a thing to be tuned by reading it, which is not what a Python string literal
    is for.
    """
    try:
        from src.utils.settings import load_settings
        filename = str((load_settings().get("system_prompts") or {})
                       .get("refine_loop", "system_prompt.refineLoop.md"))
    except Exception:  # noqa: BLE001
        filename = "system_prompt.refineLoop.md"
    if not filename.endswith(".md"):
        filename += ".md"
    path = _PROJECT_ROOT / "config" / "system_prompts" / filename
    if not path.exists():
        path = _PROJECT_ROOT / "config" / filename
    if not path.exists():
        logger.warning("refine: prompt file not found: %s", path)
        return {}
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+(.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    it = iter(parts[1:])                       # parts[0] precedes the first heading
    for name, body in zip(it, it):
        sections[name.strip()] = body.strip()
    return sections


def render_attempts(history: list) -> str:
    """What has been tried and how each one was judged, for the reviser to read.

    Without this the reviser sees one rejection at a time and oscillates: it walks
    back the very phrase it added two runs ago, because nothing told it that phrase
    had already been tried and had already failed.
    """
    if not history:
        return "(nothing yet — this is the first run)"
    lines = []
    for entry in history:
        value = " ".join(str(entry.get("value") or "").split())
        lines.append(f"run {entry.get('run')} — value: \"{value}\"")
        if entry.get("status") == "matched":
            lines.append("  judged: MET the goal")
        elif entry.get("failures"):
            for f in entry["failures"]:
                lines.append(f"  judged: missed — {f}")
        elif entry.get("summary"):
            lines.append(f"  judged: missed — {entry['summary']}")
        else:
            lines.append("  judged: missed")
    return "\n".join(lines)


def revision_messages(goal: str, target: dict, current: str, failures: list,
                      history: list) -> list[dict] | None:
    """The chat messages that ask for the next value, or None if unpromptable."""
    prompts = load_refine_prompts()
    system = prompts.get("system", "").strip()
    template = prompts.get("user", "")
    if not system or not template:
        return None
    name = target.get("title") or target.get("class_type") or "node"
    user = (template
            .replace("{{GOAL}}", str(goal or "").strip())
            .replace("{{NODE}}", f"{name} (#{target.get('node_id')})")
            .replace("{{PARAM}}", str(target.get("param") or ""))
            .replace("{{CURRENT}}", str(current or ""))
            .replace("{{FAILURES}}",
                     "\n".join(f"- {f}" for f in failures)
                     or "- (no criterion was named; the output did not meet the goal)")
            .replace("{{ATTEMPTS}}", render_attempts(history)))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_revision(reply: str) -> str:
    """The model's answer as a bare value: no fence, no label, no wrapping quotes."""
    text = str(reply or "").strip()
    fence = re.search(r"```(?:\w+)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    text = re.sub(r"^(?:new |revised |next )?(?:prompt|value)\s*[:=]\s*", "", text,
                  flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def already_tried(value: str, history: list) -> bool:
    """Whether *value* is one the loop has already spent a generation on.

    Re-running a value that has already been judged buys nothing and costs a
    render, so a repeat ends the loop as ``stalled`` rather than quietly burning
    the rest of the budget on the same picture. An empty revision counts as a
    repeat: there is no generation to be had from it either.
    """
    key = " ".join(str(value or "").split()).lower()
    if not key:
        return True
    return any(" ".join(str(e.get("value") or "").split()).lower() == key
               for e in (history or []))
