"""Deterministic brainbriefing scaffolding.

The Researcher (query_templates) LLM is good at two things: picking a template
and writing a prompt. Everything else in a brainbriefing — the input/output/prompt
node bindings, the on-disk paths, the model-existence checks — is a pure transform
over the chosen template's own structured metadata plus the staged input files.

``build_briefing_scaffold`` performs that transform in code, so the LLM never has
to hand-assemble (and mis-assemble, e.g. ``filename=None``) the mechanical fields.
It returns every field of a :class:`~src.pipeline.BrainBriefing` EXCEPT the two the
model must author: ``prompt.positive`` / ``prompt.negative`` (left empty) and
``task.description`` (passed through). Positive/negative prompt-node roles are
resolved by tracing the workflow graph — the sampler/guider's conditioning links
back to the source text-encode node — with node titles and the ComfyUI node schema
as deterministic fallbacks (no guessing).

This is generic over templates: it reads each template's own ``io``/``nodes`` and
graph, so a brand-new template with an unusual shape works without code changes.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.tools import (
    get_workflow_template,
    get_comfyui_dirs,
    check_model,
    get_node_schema,
)

# ---------------------------------------------------------------------------
# Class-name predicates (coarse, name-based — mirrors agenty_core roles.py)
# ---------------------------------------------------------------------------

def _lc(s: Any) -> str:
    return str(s or "").lower()


def _is_text_encode(cls: str) -> bool:
    c = _lc(cls)
    return ("cliptextencode" in c or "textencode" in c or "encodeprompt" in c
            or "text_encode" in c)


def _is_sampler(cls: str) -> bool:
    c = _lc(cls)
    return "sampler" in c  # KSampler, KSamplerAdvanced, SamplerCustomAdvanced, ...


def _is_guider(cls: str) -> bool:
    return "guider" in _lc(cls)  # BasicGuider, CFGGuider


# Input names that carry CONDITIONING backward through a chain toward the encoder.
_COND_INPUT_NAMES = ("positive", "negative", "conditioning", "cond", "guider",
                     "base_positive", "base_negative")

# Media-kind → agent output bucket (io.outputs already carries a mediaType).
_BUCKET_BY_MEDIA = {
    "image": "agent/images", "video": "agent/videos",
    "audio": "agent/audio", "3d": "agent/models", "model": "agent/models",
}
# Fallback by saver class when mediaType is missing.
_BUCKET_BY_CLASS = {
    "SaveImage": "agent/images", "PreviewImage": "agent/images",
    "SaveAnimatedPNG": "agent/images", "SaveAnimatedWEBP": "agent/images",
    "VHS_VideoCombine": "agent/videos", "SaveVideo": "agent/videos",
    "SaveWEBM": "agent/videos",
    "SaveAudio": "agent/audio", "VHS_SaveAudio": "agent/audio",
    "SaveAudioMP3": "agent/audio", "SaveAudioOpus": "agent/audio",
    "SaveGLB": "agent/models", "SaveGLTF": "agent/models", "Save3DModel": "agent/models",
}


def _is_loader(cls: str) -> bool:
    """Input-asset loader (LoadImage/Mask/Video/Audio, VHS_LoadImagePath, …).

    Deliberately excludes model loaders (CheckpointLoaderSimple, CLIPLoader,
    VAELoader, UNETLoader): those carry no ``image``/``video``/``audio`` payload.
    """
    c = _lc(cls)
    return "load" in c and any(t in c for t in ("image", "video", "audio", "mask"))


def _is_saver(cls: str) -> bool:
    c = _lc(cls)
    return (cls in _BUCKET_BY_CLASS or c.startswith("save")
            or "videocombine" in c or "preview" in c)


def _bucket_for(cls: str, media: str = "") -> str:
    if media in _BUCKET_BY_MEDIA:
        return _BUCKET_BY_MEDIA[media]
    if cls in _BUCKET_BY_CLASS:
        return _BUCKET_BY_CLASS[cls]
    c = _lc(cls)
    if "video" in c:
        return "agent/videos"
    if "audio" in c:
        return "agent/audio"
    if any(t in c for t in ("glb", "gltf", "3d", "mesh")):
        return "agent/models"
    return "agent/images"


def _sorted_ids(ids: list[str]) -> list[str]:
    """Sort node ids numeric-first (ComfyUI ids like '5','10','75:64')."""
    def key(nid: str):
        parts = str(nid).split(":")
        return tuple(int(p) if p.isdigit() else 10**9 for p in parts), str(nid)
    return sorted(ids, key=key)


def _unwrap(tool: Any) -> Callable:
    """Return the plain function behind a Strands ``@tool`` wrapper (or the tool)."""
    return getattr(tool, "func", tool)


def _load_json(tool: Any, *args) -> dict:
    try:
        return json.loads(_unwrap(tool)(*args))
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Graph tracing: sampler/guider conditioning links → source text-encode node
# ---------------------------------------------------------------------------

def _link_target(val: Any) -> str | None:
    """A ComfyUI API-format connection is ``[node_id, slot]``; return node_id."""
    if isinstance(val, list) and len(val) == 2 and not isinstance(val[0], (list, dict)):
        return str(val[0])
    return None


def _trace_to_text_encode(wf: dict, link: Any, seen: set | None = None,
                          depth: int = 0) -> str | None:
    """Follow a connection back through conditioning ops to a text-encode node."""
    nid = _link_target(link)
    if nid is None or depth > 8:
        return None
    seen = seen or set()
    if nid in seen:
        return None
    seen.add(nid)
    node = wf.get(nid)
    if not isinstance(node, dict):
        return None
    if _is_text_encode(node.get("class_type", "")):
        return nid
    ins = node.get("inputs", {}) or {}
    # Prefer conditioning-typed inputs; then any remaining links.
    for name in list(_COND_INPUT_NAMES) + [k for k in ins if k not in _COND_INPUT_NAMES]:
        if name in ins:
            r = _trace_to_text_encode(wf, ins[name], seen, depth + 1)
            if r:
                return r
    return None


def _title(wf: dict, nid: str) -> str:
    return _lc((wf.get(nid, {}).get("_meta", {}) or {}).get("title", ""))


def _prompt_cap(class_type: str, slot: str) -> int | None:
    """The hard character cap this model puts on this input, or None.

    Best-effort on purpose: an unknown cap must read as "no cap", never as zero.
    """
    try:
        from src.utils.model_limits import text_cap
        return text_cap(class_type, slot)
    except Exception:  # noqa: BLE001
        return None


def _resolve_prompt_nodes(wf: dict) -> tuple[str | None, str | None, list[dict], list[str]]:
    """Return (positive_id, negative_id, prompt_nodes[], warnings[]).

    Strategy (all deterministic):
      1. Trace the sampler/guider's positive/negative conditioning links back to
         the source text-encode node.
      2. Fall back to node titles ("... Negative ...").
      3. Fall back to node-schema check + first-text-encode, with a WARNING.
    """
    warnings: list[str] = []
    text_ids = [nid for nid, n in wf.items()
                if isinstance(n, dict) and _is_text_encode(n.get("class_type", ""))]

    pos_id = neg_id = None

    # 1. Graph trace from the first sampler (then its guider, if indirect).
    sampler_id = next((nid for nid, n in wf.items()
                       if isinstance(n, dict) and _is_sampler(n.get("class_type", ""))), None)
    if sampler_id is not None:
        sins = wf[sampler_id].get("inputs", {}) or {}
        pos_id = _trace_to_text_encode(wf, sins.get("positive"))
        neg_id = _trace_to_text_encode(wf, sins.get("negative"))
        if pos_id is None and "guider" in sins:  # Flux-style: sampler → guider → cond
            gid = _link_target(sins.get("guider"))
            gins = (wf.get(gid, {}) or {}).get("inputs", {}) if gid else {}
            pos_id = _trace_to_text_encode(wf, gins.get("positive") or gins.get("conditioning")
                                           or gins.get("cond"))
            neg_id = neg_id or _trace_to_text_encode(wf, gins.get("negative"))

    # 2. Title fallback.
    if pos_id is None:
        pos_by_title = [nid for nid in text_ids
                        if "posit" in _title(wf, nid) or "prompt" == _title(wf, nid)]
        pos_id = pos_by_title[0] if pos_by_title else pos_id
    if neg_id is None:
        neg_by_title = [nid for nid in text_ids if "negat" in _title(wf, nid)]
        neg_id = neg_by_title[0] if neg_by_title else neg_id

    # 3. Node-schema confirm + first-encode fallback.
    if pos_id is None and text_ids:
        # Confirm the class really takes a text input (schema check) before defaulting.
        cls = wf[text_ids[0]].get("class_type", "")
        schema = _load_json(get_node_schema, cls)
        takes_text = any("text" in k.lower() or "prompt" in k.lower()
                         for k in {**(schema.get("input_required") or {}),
                                   **(schema.get("input_optional") or {})})
        pos_id = text_ids[0]
        warnings.append(
            f"positive prompt node inferred as first text-encode node '{pos_id}' "
            f"({cls}{'' if takes_text else '; schema shows no text input'})")

    # 4. Unified-text / API models (no CLIPTextEncode): the prompt goes to a node
    #    carrying a literal 'prompt'/'text' input among the api/generator nodes.
    if pos_id is None:
        for nid, n in wf.items():
            if not isinstance(n, dict):
                continue
            ins = n.get("inputs", {}) or {}
            if any(k in ins and not isinstance(ins[k], list) for k in ("prompt", "text")):
                pos_id = nid
                break
        if pos_id is not None:
            warnings.append(f"unified-text model: prompt node resolved to '{pos_id}' "
                            f"({wf[pos_id].get('class_type','')})")

    # A single encoder feeding both conditioning inputs (e.g. negative via
    # ConditioningZeroOut off the positive encoder) is NOT a distinct negative
    # prompt node — there is no separate negative text to splice.
    if neg_id is not None and neg_id == pos_id:
        neg_id = None

    prompt_nodes: list[dict] = []
    for role, nid in (("positive", pos_id), ("negative", neg_id)):
        if nid is None:
            continue
        node = wf.get(nid, {})
        ins = node.get("inputs", {}) or {}
        slot = "text" if "text" in ins else ("prompt" if "prompt" in ins else "text")
        entry = {"node_id": nid, "role": role, "slot": slot,
                 "node": node.get("class_type", "")}
        # Tell whoever writes the prompt what it has to fit inside, while there is
        # still a prompt being written. Learning about Kling's 2,500 characters
        # after the fact costs a round trip that produces nothing.
        cap = _prompt_cap(entry["node"], slot)
        if cap:
            entry["max_chars"] = cap
        prompt_nodes.append(entry)
    return pos_id, neg_id, prompt_nodes, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_briefing_scaffold(
    template_name: str,
    staged_inputs: list[dict] | None = None,
    task_type: str = "image generation",
    task_description: str = "",
    count_iter: int = 1,
    variations: bool = False,
    resolution: dict | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """Assemble every mechanical field of a brainbriefing from *template_name*.

    Args:
        template_name: The template the Researcher selected (its one judgment).
        staged_inputs: Ordered ``[{"filename": str, "role": str}]`` for the input
            loaders, as staged by the orchestrator. ``[]`` / ``None`` for text-to-X.
        task_type:     Brainbriefing task type (e.g. ``"image generation"``).
        task_description: One-line task summary (passed through).
        count_iter:    Batch iteration count (passthrough).
        variations:    Whether each iteration uses a distinct prompt (passthrough).
        resolution:    ``{"width": int, "height": int}`` when known, else ``None``.
        verbose:       Print resolution diagnostics.

    Returns:
        A dict with the mechanical brainbriefing fields filled and
        ``prompt.positive`` empty for the LLM to author. On a fatal lookup error
        it returns a minimal ``status="blocked"`` scaffold naming the problem.
    """
    staged_inputs = staged_inputs or []
    warnings: list[str] = []
    blockers: list[str] = []

    # build_new / from-scratch: the Planner/Brain builds the graph; hand back a
    # minimal ready scaffold without a template lookup.
    if _lc(template_name) in ("build_new", "", "none"):
        return {
            "status": "ready", "blockers": [],
            "task": {"type": task_type, "description": task_description},
            "template": {"name": "build_new"},
            "input_images": [{"filename": s.get("filename", "")} for s in staged_inputs],
            "input_nodes": [], "input_image_count": len(staged_inputs),
            "output_nodes": [], "resolution_width": (resolution or {}).get("width"),
            "resolution_height": (resolution or {}).get("height"),
            "prompt": {"positive": "", "negative": None}, "prompt_nodes": [],
            "count_iter": count_iter, "variations": variations,
            "positive_prompt_node_id": None,
            "_scaffold_meta": {"positive_id": None, "negative_id": None, "warnings": []},
        }

    tpl = _load_json(get_workflow_template, template_name)
    if not tpl or tpl.get("error"):
        return {
            "status": "blocked",
            "blockers": [tpl.get("error") if tpl else f"template '{template_name}' not found"],
            "task": {"type": task_type, "description": task_description},
            "template": {"name": template_name},
            "input_images": [], "input_nodes": [], "input_image_count": 0,
            "output_nodes": [], "resolution_width": None, "resolution_height": None,
            "prompt": {"positive": "", "negative": None},
            "prompt_nodes": [], "count_iter": count_iter, "variations": variations,
            "positive_prompt_node_id": None,
        }

    resolved_name = tpl.get("name", template_name)

    # Force-build mode returns an empty canvas — treat as build_new.
    if tpl.get("build_from_scratch"):
        return build_briefing_scaffold("build_new", staged_inputs, task_type,
                                       task_description, count_iter, variations, resolution)

    io = tpl.get("io", {}) or {}
    io_inputs = io.get("inputs", []) or []
    io_outputs = io.get("outputs", []) or []
    models = tpl.get("models", []) or []
    workflow_path = tpl.get("workflow_path")

    wf: dict = {}
    if workflow_path:
        try:
            with open(workflow_path, encoding="utf-8") as fh:
                wf = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not read workflow graph ({exc}); node trace skipped")

    dirs = _load_json(get_comfyui_dirs)
    input_dir = dirs.get("input_dir", "input")
    output_dir = dirs.get("output_dir", "output")

    # Loader/saver node lists: prefer the curated io metadata, else derive from
    # the graph (local templates ship no io in the index).
    if io_inputs:
        loaders = [(str(e.get("nodeId", "")), e.get("nodeType", "LoadImage")) for e in io_inputs]
    else:
        loaders = [(nid, wf[nid].get("class_type", "LoadImage"))
                   for nid in _sorted_ids([k for k, n in wf.items()
                                           if isinstance(n, dict) and _is_loader(n.get("class_type", ""))])]
    if io_outputs:
        savers = [(str(e.get("nodeId", "")), e.get("nodeType", "SaveImage"), _lc(e.get("mediaType", "")))
                  for e in io_outputs]
    else:
        saver_ids = [k for k, n in wf.items()
                     if isinstance(n, dict) and _is_saver(n.get("class_type", ""))]
        if not saver_ids and wf:
            # Name heuristic found nothing: authoritative schema check on the
            # distinct classes (is_output_node) — catches savers we don't name.
            out_classes = {c for c in {n.get("class_type", "") for n in wf.values()
                                       if isinstance(n, dict)}
                           if _load_json(get_node_schema, c).get("is_output_node")}
            saver_ids = [k for k, n in wf.items()
                         if isinstance(n, dict) and n.get("class_type", "") in out_classes]
        savers = [(nid, wf[nid].get("class_type", "SaveImage"), "")
                  for nid in _sorted_ids(saver_ids)]
        if not savers and wf:
            warnings.append("no output/save node found in template graph "
                            "(Brain must add a saver)")

    # --- input_nodes: loaders × staged_inputs -------------------------------
    input_nodes: list[dict] = []
    for i, staged in enumerate(staged_inputs):
        nid, node_cls = loaders[i] if i < len(loaders) else (loaders[-1] if loaders else ("", "LoadImage"))
        ins = (wf.get(nid, {}) or {}).get("inputs", {}) if nid else {}
        slot = next((k for k in ins if k in ("image", "images", "video", "audio", "mask")), "image")
        fname = staged.get("filename", "")
        role = staged.get("role") or ("master_image" if i == 0 else "reference_image")
        input_nodes.append({
            "node_id": nid, "filename": fname, "role": role, "node": node_cls,
            "slot": slot, "path": f"{input_dir}/{fname}",
        })
    if len(staged_inputs) > len(loaders) and loaders:
        warnings.append(f"{len(staged_inputs)} staged inputs but template exposes "
                        f"{len(loaders)} loader(s)")

    input_images = [{"filename": s.get("filename", "")} for s in staged_inputs]

    # --- output_nodes: savers × media bucket --------------------------------
    output_nodes: list[dict] = []
    for nid, node_cls, media in savers:
        bucket = _bucket_for(node_cls, media)
        output_nodes.append({"node_id": nid, "node": node_cls,
                             "output_path": f"{output_dir}/{bucket}"})

    # --- prompt nodes: graph trace ------------------------------------------
    pos_id = neg_id = None
    prompt_nodes: list[dict] = []
    if wf:
        pos_id, neg_id, prompt_nodes, ptrace_warnings = _resolve_prompt_nodes(wf)
        warnings.extend(ptrace_warnings)

    # --- model verification -------------------------------------------------
    if models:
        res = _load_json(check_model, models)
        # check_model returns {filename: path|"False"} (or similar); flag missing.
        if isinstance(res, dict):
            for fn, val in res.items():
                if str(val).lower() in ("false", "", "none"):
                    blockers.append(f"model file not found in ComfyUI install: {fn}")

    status = "blocked" if blockers else "ready"
    # positive_prompt_node_id is only used for per-variation splicing.
    ppn_id = pos_id if (variations or count_iter > 1) else None
    # Real WARNING conditions belong in blockers (convention: status stays ready);
    # purely-diagnostic trace notes stay in _scaffold_meta only.
    blocker_list = blockers + [f"WARNING: {w}" for w in warnings if "could not read" in w]

    return {
        "status": status,
        "blockers": blocker_list,
        "task": {"type": task_type, "description": task_description},
        "template": {"name": resolved_name},
        "input_images": input_images,
        "input_nodes": input_nodes,
        "input_image_count": len(input_images),
        "output_nodes": output_nodes,
        "resolution_width": (resolution or {}).get("width"),
        "resolution_height": (resolution or {}).get("height"),
        "prompt": {"positive": "", "negative": None},
        "prompt_nodes": prompt_nodes,
        "count_iter": count_iter,
        "variations": variations,
        "positive_prompt_node_id": ppn_id,
        # Non-schema diagnostics for the merge step (dropped before validation).
        "_scaffold_meta": {"positive_id": pos_id, "negative_id": neg_id,
                           "warnings": warnings},
    }
