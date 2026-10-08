"""Where an output file goes, and what it is called.

With a context — a sequence (or asset) and a shot — every file a run produces is
saved under ComfyUI's output folder as

    <sequence>/<shot>_<suffix>_v###.<extension>

The context comes from the canvas: an AYON context node if the graph has one
(the same lookup as bEpic's "Get Path (AYON)": the last two segments of the
folder path are sequence and shot), otherwise an ``agentY context`` node. The
agent can also set or change it for part of a run (``name_outputs``), which is
how one hook writes into several shots or assets. The suffix says what the file
is ("startframe"); the version is the next free one in that folder.

No context, no change: files go to ``agent/<kind>/`` as before.

Two steps, both in the executor, so every run passes through them:

* :func:`stamp` before a graph is submitted — its savers get the prefix
  ``<sequence>/<shot>_<suffix>``;
* :func:`finalize` when a file comes back — ComfyUI's own counter
  (``…_00001_.png``) is replaced by the version.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path

logger = logging.getLogger("agentY.output")

CONTEXT_CLASS = "AgentYContext"
_SLOT = "output.context"
_STAMPED = "output.stamped"
_VERSIONS = "output.versions"
_LOCK = threading.Lock()

# Stems the builder writes when it has nothing to say about the file.
_GENERIC = {"", "image", "images", "video", "videos", "audio", "model", "models", "output",
            "comfyui", "agent", "img", "out", "result"}
_KIND_SUFFIX = {"images": "image", "videos": "video", "audio": "audio", "models": "model"}


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


def prefix_for(ctx: dict, suffix: str = "") -> str:
    """``<sequence>/<shot>_<suffix>`` (parts that are missing are left out)."""
    name = "_".join(p for p in (ctx.get("shot", ""), slug(suffix)) if p) or "output"
    return f"{ctx['sequence']}/{name}" if ctx.get("sequence") else name


def describe(ctx: dict | None = None) -> str:
    """The note the agent is given about the rule in force."""
    ctx = current() if ctx is None else ctx
    if not active(ctx):
        return ""
    example = prefix_for(ctx, "startframe") + "_v001.png"
    return (
        "[OUTPUT CONTEXT] Files are saved by rule, relative to ComfyUI's output folder: "
        "<sequence>/<shot>_<suffix>_v###.<extension>. "
        f"Now: sequence \"{ctx.get('sequence', '')}\", shot \"{ctx.get('shot', '')}\" "
        f"(from the {ctx.get('source') or 'canvas'}) - e.g. {example}. "
        "The folder, the version number and the extension are set for you; do not write "
        "them into a filename_prefix yourself. What you supply is the SUFFIX: before you "
        "run a workflow, call name_outputs(suffix=\"…\", workflow_path=\"…\") with one or "
        "two words for what the file IS in this job (startframe, endframe, hero_ref, "
        "plate_clean). When one request produces files for SEVERAL shots, sequences or "
        "assets (\"the male lead and the woman\"), name each workflow for its own: "
        "name_outputs(suffix=\"ref\", shot=\"male_lead\", workflow_path=…), then "
        "shot=\"woman\" for the other. In a batch of variants, give the saver's "
        "filename_prefix one value per variant in the form \"<sequence>/<shot>_<suffix>\". "
        "Report files by the path the run returns.]")


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
    """A prefix agentY routed (``agent/…``) or nobody chose (no folder at all)."""
    p = str(prefix).replace("\\", "/").lstrip("./")
    return "/" not in p or p.startswith("agent/")


def set_prefixes(graph: dict, ctx: dict, suffix: str = "", node_ids=None) -> dict:
    """Name the savers of *graph* for *ctx*; returns ``{node_id: prefix}``.

    A saver whose prefix is wired, or already names a folder of its own, is left
    alone unless *suffix* is given — then the caller is naming this graph's
    outputs outright.
    """
    done = {}
    wanted = {str(n) for n in node_ids} if node_ids is not None else None
    for nid, node in (graph or {}).items():
        if not _is_saver(node) or (wanted is not None and str(nid) not in wanted):
            continue
        old = node["inputs"]["filename_prefix"]
        if not suffix and not _ours(old):
            continue
        new = prefix_for(ctx, suffix or ctx.get("suffix") or _own_suffix(old, node.get("class_type", "")))
        node["inputs"]["filename_prefix"] = new
        done[str(nid)] = new
    return done


def stamp(graph: dict) -> dict:
    """Before submission: apply the rule to *graph* and note what to version."""
    ctx = current()
    if not active(ctx) or not isinstance(graph, dict):
        return {}
    done = set_prefixes(graph, ctx)
    try:
        stamped = _scope().slot(_STAMPED, set)
        for node in graph.values():
            if _is_saver(node):
                p = str(node["inputs"]["filename_prefix"]).replace("\\", "/").lstrip("./")
                if not p.startswith("agent/"):
                    stamped.add(p)
    except Exception:  # noqa: BLE001
        pass
    return done


# ── a file that came back ───────────────────────────────────────────────────

def _next_version(folder: Path, base: str) -> int:
    seen = [0]
    pattern = re.compile(re.escape(base) + r"_v(\d{3,})\.[A-Za-z0-9]+$", re.I)
    try:
        for name in os.listdir(folder):
            m = pattern.match(name)
            if m:
                seen.append(int(m.group(1)))
    except OSError:
        pass
    return max(seen) + 1


def finalize(path) -> Path:
    """Give a saved file its version: ``shot_sfx_00001_.png`` -> ``shot_sfx_v003.png``.

    Only files whose prefix :func:`stamp` set this turn; anything else is handed
    back untouched. Files of one save (a video and its preview frame) share a
    version. Never raises: a file that cannot be renamed keeps its name.
    """
    p = Path(str(path))
    try:
        scope = _scope()
        stamped = scope.slot(_STAMPED, set)
        if not stamped:
            return p
        folder = p.parent.as_posix().rstrip("/")
        for prefix in sorted(stamped, key=len, reverse=True):
            sub, _, base = prefix.rpartition("/")
            if sub and not folder.lower().endswith("/" + sub.lower()):
                continue
            m = re.match(re.escape(base) + r"_(\d{3,})_?(\.[A-Za-z0-9]+)$", p.name)
            if not m:
                continue
            with _LOCK:
                versions = scope.slot(_VERSIONS, dict)
                key = (folder.lower(), base, m.group(1))
                n = versions.get(key) or _next_version(p.parent, base)
                while True:
                    target = p.with_name(f"{base}_v{n:03d}{m.group(2)}")
                    if not target.exists():
                        break
                    if key in versions:      # the version is taken for this type already
                        return p
                    n += 1
                versions[key] = n
                os.replace(p, target)
            logger.info("output: %s -> %s", p.name, target.name)
            return target
    except Exception as exc:  # noqa: BLE001
        logger.warning("output: could not version %s (%s)", p, exc)
    return p
