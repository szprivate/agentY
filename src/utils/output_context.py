"""Where an output file goes, and what it is called.

With a context — a sequence (or asset) and a shot — every file a run produces is
saved under ComfyUI's output folder in bEpic's layout, the one ``bepicSetPath`` /
``Get Path (AYON)`` in bepic_templates build:

    <Sequence>/<Shot>/images/v###/<shot>_v###_<suffix>      (+ ComfyUI's counter)
    <Sequence>/<Shot>/videos/v###/<shot>_v###_<suffix>

The context comes from the canvas: an AYON context node if the graph has one
(the last two segments of its folder path are sequence and shot, as in Get Path
(AYON)), otherwise an ``agentY context`` node. The agent can also set or change
it for part of a run (``name_outputs``), which is how one request writes into
several shots or assets. The suffix says what the file is ("startframe"); the
version is the next one that has no such file yet.

The whole name is written into the save node BEFORE the graph runs
(:func:`stamp`, called by the executor on every submission). Nothing is renamed
afterwards, so the file ComfyUI saved is the file its history shows.

No context, no change: files go to ``agent/<kind>/`` as before.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger("agentY.output")

CONTEXT_CLASS = "AgentYContext"
_SLOT = "output.context"
_RESERVED = "output.versions"

# Stems the builder writes when it has nothing to say about the file.
_GENERIC = {"", "image", "images", "video", "videos", "audio", "model", "models", "output",
            "comfyui", "agent", "img", "out", "result"}
_KIND_SUFFIX = {"images": "image", "videos": "video", "audio": "audio", "models": "model"}
KINDS = ("images", "videos", "audio", "models", "mattes")


def slug(text, fallback: str = "") -> str:
    """A name safe in a file name: letters, digits, ``-`` and ``_``."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text or "").strip()).strip("_")
    return re.sub(r"_+", "_", s)[:60] or fallback


# ── reading the canvas ──────────────────────────────────────────────────────

def _folder_path(d):
    return (d.get("folderPath") or d.get("folder_path")) if isinstance(d, dict) else None


def ayon_folder_path(ctx) -> str | None:
    """The folder path in a parsed AYON context (publish or context-node shape)."""
    if not isinstance(ctx, dict):
        return None
    instances = ctx.get("instances")
    if isinstance(instances, list):
        with_folder = [i for i in instances if _folder_path(i)]
        for inst in with_folder:
            if inst.get("active"):
                return _folder_path(inst)
        if with_folder:
            return _folder_path(with_folder[0])
    return _folder_path(ctx) or _folder_path(ctx.get("context"))


def split_folder_path(folder_path) -> tuple[str, str]:
    """``/03_sequences/spec/spec_0210`` -> ``("spec", "spec_0210")``."""
    segs = [s for s in str(folder_path).replace("\\", "/").split("/") if s]
    if len(segs) >= 2:
        return segs[-2], segs[-1]
    return ("", segs[0]) if segs else ("", "")


def _ayon_context(prompt: dict) -> dict | None:
    fallback = None
    for node in prompt.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        class_type = str(node.get("class_type", "")).lower()
        if class_type == CONTEXT_CLASS.lower():
            continue
        for val in node["inputs"].values():
            if not isinstance(val, str) or "{" not in val:
                continue
            try:
                parsed = json.loads(val)
            except (ValueError, TypeError):
                continue
            if ayon_folder_path(parsed) is None:
                continue
            if "ayon" in class_type or "context" in class_type:
                return parsed
            fallback = fallback or parsed
    return fallback


def from_canvas(prompt: dict | None) -> dict:
    """``{sequence, shot, source}`` for the graph, or ``{}`` when it names none."""
    if not isinstance(prompt, dict):
        return {}
    ayon = _ayon_context(prompt)
    if ayon is not None:
        seq, shot = split_folder_path(ayon_folder_path(ayon))
        if shot:
            return {"sequence": slug(seq), "shot": slug(shot), "source": "AYON context node"}
    for nid, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") == CONTEXT_CLASS:
            inputs = node.get("inputs") or {}
            seq, shot = inputs.get("sequence"), inputs.get("shot")
            seq = slug(seq) if isinstance(seq, str) else ""
            shot = slug(shot) if isinstance(shot, str) else ""
            if seq or shot:
                return {"sequence": seq, "shot": shot, "source": f"agentY context node {nid}"}
    return {}


# ── the turn's context ──────────────────────────────────────────────────────

def _scope():
    from agenty_core.utils import turn_scope
    return turn_scope.current()


def current() -> dict:
    """The context in force right now (``{}`` = none)."""
    try:
        return dict(_scope().slot(_SLOT, dict))
    except Exception:  # noqa: BLE001
        return {}


def set_current(sequence=None, shot=None, suffix=None, source: str = "") -> dict:
    """Set or change parts of the context; ``None`` leaves a part as it is."""
    ctx = _scope().slot(_SLOT, dict)
    if sequence is not None:
        ctx["sequence"] = slug(sequence)
    if shot is not None:
        ctx["shot"] = slug(shot)
    if suffix is not None:
        ctx["suffix"] = slug(suffix)
    if source:
        ctx["source"] = source
    return dict(ctx)


def active(ctx: dict | None = None) -> bool:
    ctx = current() if ctx is None else ctx
    return bool(ctx.get("shot") or ctx.get("sequence"))


# ── the layout ──────────────────────────────────────────────────────────────

def shot_root(ctx: dict) -> str:
    """``<Sequence>/<Shot>`` - the shot's folder under the output folder."""
    return "/".join(p for p in (ctx.get("sequence", ""), ctx.get("shot", "")) if p)


def _file_stem(ctx: dict) -> str:
    return (ctx.get("shot") or ctx.get("sequence") or "output").lower()


def prefix_for(ctx: dict, suffix: str, kind: str = "images", version: int = 1) -> str:
    """bEpic's path: ``<Seq>/<Shot>/<kind>/v###/<shot>_v###_<suffix>``."""
    v = f"v{int(version):03d}"
    return f"{shot_root(ctx)}/{kind}/{v}/{_file_stem(ctx)}_{v}_{slug(suffix, 'output')}"


def _output_root():
    try:
        from src.utils.hook_cache import output_dir
        return output_dir()
    except Exception:  # noqa: BLE001
        return None


def next_version(ctx: dict, suffix: str, root=None) -> int:
    """The first version that has no ``<shot>_v###_<suffix>`` file yet, in any
    kind folder, and that nothing else in this turn has taken."""
    root = _output_root() if root is None else root
    name = re.compile(re.escape(f"{_file_stem(ctx)}_v") + r"(\d{3,})_"
                      + re.escape(slug(suffix, "output")) + r"(?:[_.]|$)", re.I)
    used = [0]
    if root is not None:
        base = os.path.join(str(root), *shot_root(ctx).split("/"))
        for kind in KINDS:
            try:
                versions = os.listdir(os.path.join(base, kind))
            except OSError:
                continue
            for vdir in versions:
                try:
                    files = os.listdir(os.path.join(base, kind, vdir))
                except OSError:
                    continue
                used += [int(m.group(1)) for m in map(name.match, files) if m]
    key = (shot_root(ctx).lower(), slug(suffix, "output").lower())
    try:
        taken = _scope().slot(_RESERVED, dict)
        n = max(max(used), taken.get(key, 0)) + 1
        taken[key] = n
    except Exception:  # noqa: BLE001
        n = max(used) + 1
    return n


def describe(ctx: dict | None = None) -> str:
    """The note the agent is given about the rule in force."""
    ctx = current() if ctx is None else ctx
    if not active(ctx):
        return ""
    example = prefix_for(ctx, "startframe", "images", 1) + "_00001_.png"
    return (
        "[OUTPUT CONTEXT] Files are saved by rule, relative to ComfyUI's output folder: "
        "<Sequence>/<Shot>/<images|videos>/v###/<shot>_v###_<suffix>. "
        f"Now: sequence \"{ctx.get('sequence', '')}\", shot \"{ctx.get('shot', '')}\" "
        f"(from the {ctx.get('source') or 'canvas'}) - e.g. {example}. "
        "The folders and the version number are set for you, in the save node, before the "
        "run; never write a filename_prefix for these yourself. What you supply is the "
        "SUFFIX: before you run a workflow, call name_outputs(suffix=\"…\", "
        "workflow_path=\"…\") with one or two words for what the file IS in this job "
        "(startframe, endframe, hero_ref, plate_clean). When one request produces files "
        "for SEVERAL shots, sequences or assets (\"the male lead and the woman\"), name "
        "each workflow for its own: name_outputs(suffix=\"ref\", shot=\"male_lead\", "
        "workflow_path=…), then shot=\"woman\" for the other. Report files by the path "
        "the run returns.]")


# ── a graph about to run ────────────────────────────────────────────────────

def _is_saver(node: dict) -> bool:
    return isinstance(node, dict) and isinstance((node.get("inputs") or {}).get("filename_prefix"), str)


def _kind(class_type: str) -> str:
    try:
        from agenty_core.tools.comfyui import _agent_media_bucket
        return _agent_media_bucket(class_type)
    except Exception:  # noqa: BLE001
        return "images"


def _own_suffix(prefix: str, class_type: str) -> str:
    """What the builder called this output, when it called it anything."""
    stem = slug(str(prefix).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
    return _KIND_SUFFIX.get(_kind(class_type), "image") if stem.lower() in _GENERIC else stem


def _ours(prefix: str) -> bool:
    """A prefix agentY routed (``agent/…``) or nobody chose (no folder at all).

    Anything else names a folder of its own - the user's, or one this rule
    already wrote - and is left exactly as it is.
    """
    p = str(prefix).replace("\\", "/")
    p = p[2:] if p.startswith("./") else p
    return "/" not in p or p.startswith("agent/")


def set_prefixes(graph: dict, ctx: dict, suffix: str = "", node_ids=None, root=None) -> dict:
    """Name the savers of *graph* for *ctx*; returns ``{node_id: prefix}``.

    A saver whose prefix is wired, or already names a folder of its own, is left
    alone unless *suffix* is given — then the caller is naming this graph's
    outputs outright. Savers named in one call share a version, so the image
    and the video of one generation carry the same number.
    """
    done, versions = {}, {}
    wanted = {str(n) for n in node_ids} if node_ids is not None else None
    for nid, node in (graph or {}).items():
        if not _is_saver(node) or (wanted is not None and str(nid) not in wanted):
            continue
        old = node["inputs"]["filename_prefix"]
        if not suffix and not _ours(old):
            continue
        sfx = slug(suffix or ctx.get("suffix") or _own_suffix(old, node.get("class_type", "")), "output")
        if sfx not in versions:
            versions[sfx] = next_version(ctx, sfx, root)
        new = prefix_for(ctx, sfx, _kind(node.get("class_type", "")), versions[sfx])
        node["inputs"]["filename_prefix"] = new
        done[str(nid)] = new
    return done


def stamp(graph: dict) -> dict:
    """Before submission: write the rule's names into the savers of *graph*."""
    ctx = current()
    if not active(ctx) or not isinstance(graph, dict):
        return {}
    return set_prefixes(graph, ctx)
