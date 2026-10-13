"""graph_splice - the common changes to a graph, worked out in code.

Adding a ControlNet, an image to start from, a hires pass, a patch on the model:
each is a handful of nodes and wires that are the same every time, and where they
attach can be read off the graph by TYPE - the sampler is the node that takes a
model, two conditionings and a latent; the VAE is whatever the decoder reads.
A model asked to do it by hand picks the right nodes and then loses a wire
(measured with small local orchestrators, 2026-10-13: every failed baseline case
was one of these, with a required input left unfed).

So the model says WHAT - ``splice(graph, "controlnet", {...})`` - and this module
says how: a list of :mod:`canvas_edit` operations (add / connect), values to set
on nodes that are already there, and nodes that are no longer read. Pure: no
I/O, no pipeline. The caller applies the operations through ``canvas_edit.plan``
- to the open canvas or to a workflow file, the graph is the same API format.

Nothing here guesses. An operation that cannot find what it attaches to (no
sampler, two candidates it cannot tell apart) answers ``ok: False`` and says
what it looked for, and the caller falls back to editing by hand.
"""

from __future__ import annotations

OPERATIONS = {
    "image_to_image": "start from an image instead of an empty latent "
                      "(params: image = file name in ComfyUI's input folder, denoise = 0..1)",
    "controlnet": "guide the picture with a control image "
                  "(params: control_net_name, image, preprocessor = 'canny' or 'none', strength)",
    "patch_model": "put a model patch in front of the sampler - FreeU_V2, LoraLoaderModelOnly, "
                   "ModelSamplingAuraFlow, any node that takes a MODEL and gives one "
                   "(params: class_type, plus that node's own values)",
    "lora": "add a LoRA on model and text encoder (params: lora_name, strength_model, strength_clip)",
    "hires_fix": "a second sampling pass on the upscaled latent (params: scale_by, denoise, steps)",
    "model_upscale": "upscale the decoded picture with an upscale model (params: model_name)",
    "preview_output": "show the picture instead of saving it (replaces SaveImage with PreviewImage)",
    "save_output": "save the picture instead of only showing it (replaces PreviewImage with SaveImage; "
                   "params: filename_prefix)",
}


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def _spec_type(spec):
    return spec[0] if isinstance(spec, (list, tuple)) and spec else None


def _inputs(info: dict, cls: str) -> dict:
    inp = (info.get(cls) or {}).get("input") or {}
    out = dict(inp.get("optional") or {})
    out.update(inp.get("required") or {})
    return out


def _link_inputs(info: dict, cls: str, kind: str) -> list:
    """Names of the inputs of *cls* that take a wire of type *kind*."""
    return [n for n, s in _inputs(info, cls).items() if _spec_type(s) == kind]


def _out_slot(info: dict, cls: str, kind: str):
    outs = (info.get(cls) or {}).get("output") or []
    return outs.index(kind) if kind in outs else None


def _src(graph: dict, node_id: str, name: str):
    v = ((graph.get(str(node_id)) or {}).get("inputs") or {}).get(name)
    return (str(v[0]), int(v[1])) if _is_link(v) else None


def _consumers(graph: dict, src_id: str, slot: int) -> list:
    out = []
    for nid, node in graph.items():
        for name, v in ((node or {}).get("inputs") or {}).items():
            if _is_link(v) and str(v[0]) == str(src_id) and int(v[1]) == int(slot):
                out.append((str(nid), name))
    return out


def _upstream(graph: dict, node_id: str, seen=None) -> set:
    seen = seen if seen is not None else set()
    for v in ((graph.get(str(node_id)) or {}).get("inputs") or {}).values():
        if _is_link(v) and str(v[0]) not in seen and str(v[0]) in graph:
            seen.add(str(v[0]))
            _upstream(graph, str(v[0]), seen)
    return seen


def samplers(graph: dict, info: dict) -> list:
    """The sampler nodes, first pass first: a node that takes a MODEL, a positive
    and a negative CONDITIONING and a LATENT, and gives a LATENT."""
    found = []
    for nid, node in graph.items():
        cls = (node or {}).get("class_type")
        if (_link_inputs(info, cls, "MODEL") and len(_link_inputs(info, cls, "CONDITIONING")) >= 2
                and _link_inputs(info, cls, "LATENT") and _out_slot(info, cls, "LATENT") is not None):
            found.append(str(nid))
    # Upstream first: a second pass has the first one somewhere behind it.
    return sorted(found, key=lambda n: (len(_upstream(graph, n) & set(found)), int(n) if n.isdigit() else 0))


def _cond_inputs(info: dict, cls: str) -> tuple:
    names = _link_inputs(info, cls, "CONDITIONING")
    pos = next((n for n in names if "pos" in n.lower()), names[0])
    neg = next((n for n in names if "neg" in n.lower()), names[1] if len(names) > 1 else names[0])
    return pos, neg


def _find_output(graph: dict, info: dict, kind: str, near: str | None = None):
    """A node output of type *kind*: the one *near*'s ancestry already uses, else
    the only one in the graph, else None."""
    cands = []
    for nid, node in graph.items():
        slot = _out_slot(info, (node or {}).get("class_type"), kind)
        if slot is not None:
            cands.append((str(nid), slot))
    if near is not None:
        up = _upstream(graph, near) | {str(near)}
        close = [c for c in cands if c[0] in up]
        if len(close) == 1:
            return close[0]
        cands = close or cands
    return cands[0] if len(cands) == 1 else None


def _vae(graph: dict, info: dict, near: str | None = None):
    """Where the graph's VAE comes from: what a decoder reads, else the only VAE output."""
    for nid, node in graph.items():
        cls = (node or {}).get("class_type")
        if _out_slot(info, cls, "IMAGE") is not None and _link_inputs(info, cls, "LATENT"):
            for name in _link_inputs(info, cls, "VAE"):
                got = _src(graph, nid, name)
                if got:
                    return got
    return _find_output(graph, info, "VAE", near)


def _snap(info: dict, cls: str, name: str, value):
    """*value* as one of the options of a list input - the exact one, else the one
    whose name contains it, else (nothing asked for) the first. None = no such option."""
    spec = _inputs(info, cls).get(name)
    t = _spec_type(spec)
    opts = t if isinstance(t, list) else ((spec[1].get("options") if len(spec) > 1 and isinstance(spec[1], dict)
                                           else None) if t == "COMBO" else None)
    if not isinstance(opts, list):
        return value
    if value in (None, ""):
        return opts[0] if opts else None
    if value in opts:
        return value
    want = str(value).lower().replace("\\", "/")
    for o in opts:
        s = str(o).lower().replace("\\", "/")
        if s.endswith("/" + want) or s == want or s.rsplit("/", 1)[-1].rsplit(".", 1)[0] == want.rsplit(".", 1)[0]:
            return o
    hits = [o for o in opts if want.rsplit(".", 1)[0] in str(o).lower()]
    return hits[0] if len(hits) == 1 else None


def _fail(msg: str) -> dict:
    return {"ok": False, "error": msg}


def _need(info: dict, *classes) -> str:
    missing = [c for c in classes if c not in info or not info.get(c)]
    return ("not installed in this ComfyUI: " + ", ".join(missing)) if missing else ""


def _decoder_after(graph: dict, info: dict, sampler: str):
    """(node, input) pairs reading the sampler's LATENT output."""
    slot = _out_slot(info, graph[sampler]["class_type"], "LATENT")
    return _consumers(graph, sampler, slot), slot


# ── the operations ──────────────────────────────────────────────────────────────

def _image_to_image(graph, info, p):
    miss = _need(info, "LoadImage", "VAEEncode")
    if miss:
        return _fail(miss)
    found = samplers(graph, info)
    if not found:
        return _fail("no sampler in the graph to give a starting image to")
    first = found[0]
    lat = _link_inputs(info, graph[first]["class_type"], "LATENT")[0]
    vae = _vae(graph, info, first)
    if not vae:
        return _fail("the graph has no VAE to encode the image with (or more than one, and none is read by a decoder)")
    image = _snap(info, "LoadImage", "image", p.get("image"))
    if image is None:
        return _fail(f"no image named {p.get('image')!r} in ComfyUI's input folder - upload it first (upload_image)")
    ops = [{"op": "add", "class_type": "LoadImage", "ref": "i2i_image", "params": {"image": image}, "near": first},
           {"op": "add", "class_type": "VAEEncode", "ref": "i2i_encode", "near": first},
           {"op": "connect", "from": "i2i_image", "output": 0, "to": "i2i_encode", "input": "pixels"},
           {"op": "connect", "from": vae[0], "output": vae[1], "to": "i2i_encode", "input": "vae"},
           {"op": "connect", "from": "i2i_encode", "output": 0, "to": first, "input": lat}]
    out = {"ok": True, "ops": ops, "set": {}, "delete": [], "summary": f"image-to-image from {image}"}
    old = _src(graph, first, lat)
    if old and len(_consumers(graph, *old)) == 1 and not _upstream(graph, old[0]):
        out["delete"].append(old[0])                 # the empty latent nothing else reads
    if "denoise" in _inputs(info, graph[first]["class_type"]):
        out["set"][first] = {"denoise": float(p.get("denoise", 0.6))}
        out["summary"] += f", denoise {out['set'][first]['denoise']}"
    return out


def _controlnet(graph, info, p):
    miss = _need(info, "ControlNetLoader", "ControlNetApplyAdvanced", "LoadImage")
    if miss:
        return _fail(miss)
    found = samplers(graph, info)
    if not found:
        return _fail("no sampler in the graph to guide")
    first = found[0]
    pos, neg = _cond_inputs(info, graph[first]["class_type"])
    s_pos, s_neg = _src(graph, first, pos), _src(graph, first, neg)
    if not s_pos or not s_neg:
        return _fail(f"the sampler (node {first}) has no positive / negative conditioning wired yet")
    name = _snap(info, "ControlNetLoader", "control_net_name", p.get("control_net_name") or p.get("model"))
    if name is None:
        return _fail(f"no ControlNet model matching {p.get('control_net_name') or p.get('model')!r} is installed "
                     "(check_model lists what is)")
    image = _snap(info, "LoadImage", "image", p.get("image"))
    if image is None:
        return _fail(f"no image named {p.get('image')!r} in ComfyUI's input folder - upload it first (upload_image)")
    pre = str(p.get("preprocessor") or "none").strip().lower()
    ops = [{"op": "add", "class_type": "ControlNetLoader", "ref": "cn_loader",
            "params": {"control_net_name": name}, "near": first},
           {"op": "add", "class_type": "LoadImage", "ref": "cn_image", "params": {"image": image}, "near": first}]
    img_from = "cn_image"
    if pre == "canny":
        if _need(info, "Canny"):
            return _fail(_need(info, "Canny"))
        ops += [{"op": "add", "class_type": "Canny", "ref": "cn_canny", "near": first,
                 "params": {k: float(p[k]) for k in ("low_threshold", "high_threshold") if k in p}},
                {"op": "connect", "from": "cn_image", "output": 0, "to": "cn_canny", "input": "image"}]
        img_from = "cn_canny"
    elif pre not in ("none", ""):
        return _fail(f"preprocessor {pre!r} is not one this operation adds - use 'canny' or 'none' and add the "
                     "preprocessor node yourself")
    params = {k: float(p[k]) for k in ("strength", "start_percent", "end_percent") if k in p}
    ops += [{"op": "add", "class_type": "ControlNetApplyAdvanced", "ref": "cn_apply", "params": params, "near": first},
            {"op": "connect", "from": s_pos[0], "output": s_pos[1], "to": "cn_apply", "input": "positive"},
            {"op": "connect", "from": s_neg[0], "output": s_neg[1], "to": "cn_apply", "input": "negative"},
            {"op": "connect", "from": "cn_loader", "output": 0, "to": "cn_apply", "input": "control_net"},
            {"op": "connect", "from": img_from, "output": 0, "to": "cn_apply", "input": "image"}]
    for s in found:                                   # every pass reads the guided conditioning
        sp, sn = _cond_inputs(info, graph[s]["class_type"])
        if _src(graph, s, sp) == s_pos:
            ops.append({"op": "connect", "from": "cn_apply", "output": 0, "to": s, "input": sp})
        if _src(graph, s, sn) == s_neg:
            ops.append({"op": "connect", "from": "cn_apply", "output": 1, "to": s, "input": sn})
    return {"ok": True, "ops": ops, "set": {}, "delete": [],
            "summary": f"ControlNet {name} on {image}" + (" through Canny" if pre == "canny" else "")}


def _patch(graph, info, p, cls=None, kinds=("MODEL",)):
    cls = str(cls or p.get("class_type") or "")
    if not cls:
        return _fail("patch_model needs class_type - the node to put in front of the sampler, e.g. FreeU_V2")
    if _need(info, cls):
        return _fail(_need(info, cls))
    found = samplers(graph, info)
    if not found:
        return _fail("no sampler in the graph to put a patch in front of")
    spec = _inputs(info, cls)
    values = {}
    for k, v in p.items():
        if k in ("class_type",) or k not in spec:
            continue
        snapped = _snap(info, cls, k, v)
        if snapped is None:
            return _fail(f"{cls}.{k}: nothing matching {v!r} is installed")
        values[k] = snapped
    for k, s in spec.items():                         # a required list with nothing asked for: its first option
        t = _spec_type(s)
        if k not in values and (isinstance(t, list) or t == "COMBO") and k in (
                ((info.get(cls) or {}).get("input") or {}).get("required") or {}):
            first_opt = _snap(info, cls, k, None)
            if first_opt is not None:
                values[k] = first_opt
    ops = [{"op": "add", "class_type": cls, "ref": "patch", "params": values, "near": found[0]}]
    wired = 0
    for kind in kinds:
        ins = _link_inputs(info, cls, kind)
        out_slot = _out_slot(info, cls, kind)
        if not ins or out_slot is None:
            if kind == "MODEL":
                return _fail(f"{cls} does not take a {kind} and give one, so it is not a patch on the model")
            continue
        if kind == "MODEL":
            readers = [(s, _link_inputs(info, graph[s]["class_type"], "MODEL")[0]) for s in found]
        else:                                         # CLIP: the text encoders
            readers = [(nid, name) for nid, node in graph.items()
                       for name in _link_inputs(info, (node or {}).get("class_type"), kind)
                       if _out_slot(info, node.get("class_type"), "CONDITIONING") is not None]
        sources = {_src(graph, n, name) for n, name in readers} - {None}
        if len(sources) != 1:
            if kind == "MODEL":
                return _fail(f"the samplers read their model from {len(sources)} places - which one the patch "
                             "belongs on cannot be told from the graph")
            continue
        source = next(iter(sources))
        ops.append({"op": "connect", "from": source[0], "output": source[1], "to": "patch", "input": ins[0]})
        for n, name in readers:
            ops.append({"op": "connect", "from": "patch", "output": out_slot, "to": n, "input": name})
        wired += 1
    if not wired:
        return _fail(f"{cls} could not be wired in")
    return {"ok": True, "ops": ops, "set": {}, "delete": [], "summary": f"{cls} in front of the sampler"}


def _lora(graph, info, p):
    return _patch(graph, info, {k: v for k, v in p.items() if k != "class_type"}, cls="LoraLoader",
                  kinds=("MODEL", "CLIP"))


def _hires_fix(graph, info, p):
    miss = _need(info, "LatentUpscaleBy")
    if miss:
        return _fail(miss)
    found = samplers(graph, info)
    if not found:
        return _fail("no sampler in the graph to add a second pass after")
    last = found[-1]
    node = graph[last]
    cls = node["class_type"]
    readers, slot = _decoder_after(graph, info, last)
    if not readers:
        return _fail(f"nothing reads the sampler's result (node {last}), so there is nowhere to put a second pass")
    lat = _link_inputs(info, cls, "LATENT")[0]
    values = {k: v for k, v in (node.get("inputs") or {}).items() if not _is_link(v)}
    if "denoise" in _inputs(info, cls):
        values["denoise"] = float(p.get("denoise", 0.5))
    if "steps" in p and "steps" in values:
        values["steps"] = int(p["steps"])
    up = {"scale_by": float(p.get("scale_by", p.get("scale", 1.5)))}
    if p.get("upscale_method"):
        up["upscale_method"] = p["upscale_method"]
    ops = [{"op": "add", "class_type": "LatentUpscaleBy", "ref": "hires_up", "params": up, "near": last},
           {"op": "connect", "from": last, "output": slot, "to": "hires_up", "input": "samples"},
           {"op": "add", "class_type": cls, "ref": "hires_pass", "params": values, "near": last}]
    for name, v in (node.get("inputs") or {}).items():
        if _is_link(v) and name != lat:
            ops.append({"op": "connect", "from": str(v[0]), "output": int(v[1]), "to": "hires_pass", "input": name})
    ops.append({"op": "connect", "from": "hires_up", "output": 0, "to": "hires_pass", "input": lat})
    for n, name in readers:
        ops.append({"op": "connect", "from": "hires_pass", "output": slot, "to": n, "input": name})
    return {"ok": True, "ops": ops, "set": {}, "delete": [],
            "summary": f"hires fix: x{up['scale_by']}, second pass at denoise {values.get('denoise', '-')}"}


def _image_readers(graph, info):
    """(source, [(node, input)]) for the picture that reaches the output nodes."""
    by_src: dict = {}
    for nid, node in graph.items():
        cls = (node or {}).get("class_type")
        if not (info.get(cls) or {}).get("output_node"):
            continue
        for name in _link_inputs(info, cls, "IMAGE"):
            got = _src(graph, nid, name)
            if got:
                by_src.setdefault(got, []).append((str(nid), name))
    return by_src


def _model_upscale(graph, info, p):
    miss = _need(info, "UpscaleModelLoader", "ImageUpscaleWithModel")
    if miss:
        return _fail(miss)
    by_src = _image_readers(graph, info)
    if len(by_src) != 1:
        return _fail("the picture that is saved could not be told: "
                     + ("no output node reads an image" if not by_src else "the output nodes read different images"))
    source, readers = next(iter(by_src.items()))
    name = _snap(info, "UpscaleModelLoader", "model_name", p.get("model_name") or p.get("model"))
    if name is None:
        return _fail(f"no upscale model matching {p.get('model_name') or p.get('model')!r} is installed "
                     "(check_model lists what is)")
    ops = [{"op": "add", "class_type": "UpscaleModelLoader", "ref": "up_loader", "params": {"model_name": name},
            "near": source[0]},
           {"op": "add", "class_type": "ImageUpscaleWithModel", "ref": "up_apply", "near": source[0]},
           {"op": "connect", "from": "up_loader", "output": 0, "to": "up_apply", "input": "upscale_model"},
           {"op": "connect", "from": source[0], "output": source[1], "to": "up_apply", "input": "image"}]
    for n, inp in readers:
        ops.append({"op": "connect", "from": "up_apply", "output": 0, "to": n, "input": inp})
    return {"ok": True, "ops": ops, "set": {}, "delete": [], "summary": f"upscale with {name} before the output"}


def _swap_output(graph, info, p, old, new):
    miss = _need(info, new)
    if miss:
        return _fail(miss)
    nodes = [str(n) for n, node in graph.items() if (node or {}).get("class_type") == old]
    if not nodes:
        return _fail(f"there is no {old} node in the graph to replace")
    ops, gone = [], []
    for i, nid in enumerate(nodes):
        got = _src(graph, nid, "images")
        if not got:
            continue
        ref = f"out_{i}"
        params = {"filename_prefix": str(p["filename_prefix"])} if new == "SaveImage" and p.get("filename_prefix") else {}
        ops += [{"op": "add", "class_type": new, "ref": ref, "params": params, "near": nid},
                {"op": "connect", "from": got[0], "output": got[1], "to": ref, "input": "images"}]
        gone.append(nid)
    if not ops:
        return _fail(f"the {old} node has no picture wired into it")
    return {"ok": True, "ops": ops, "set": {}, "delete": gone, "summary": f"{new} in place of {old}"}


_RUN = {
    "image_to_image": _image_to_image,
    "controlnet": _controlnet,
    "patch_model": _patch,
    "lora": _lora,
    "hires_fix": _hires_fix,
    "model_upscale": _model_upscale,
    "preview_output": lambda g, i, p: _swap_output(g, i, p, "SaveImage", "PreviewImage"),
    "save_output": lambda g, i, p: _swap_output(g, i, p, "PreviewImage", "SaveImage"),
}


def splice(graph: dict, operation: str, params: dict | None, object_info) -> dict:
    """Work out *operation* on *graph*.

    Returns ``{"ok": True, "ops": [...], "set": {node: {input: value}},
    "delete": [node ids], "summary": str}`` - or ``{"ok": False, "error": str}``.
    *object_info* is anything with ``.get(class_type)``.
    """
    name = str(operation or "").strip().lower().replace("-", "_").replace(" ", "_")
    run = _RUN.get(name)
    if run is None:
        return {"ok": False, "error": f"unknown operation {operation!r}",
                "operations": dict(OPERATIONS)}
    if not isinstance(graph, dict) or not graph:
        return _fail("there is no graph to change")
    try:
        return run(graph, _Info(object_info), dict(params or {}))
    except Exception as exc:  # noqa: BLE001 - a graph this cannot read is edited by hand
        return _fail(f"the graph could not be read for this operation ({type(exc).__name__}: {exc})")


class _Info:
    """``object_info`` with ``in`` and a cache, whatever kind of mapping it is."""

    def __init__(self, source) -> None:
        self._source, self._seen = source, {}

    def get(self, cls, default=None):
        if cls not in self._seen:
            try:
                self._seen[cls] = self._source.get(cls) if cls else None
            except Exception:  # noqa: BLE001
                self._seen[cls] = None
        return self._seen[cls] or default

    def __contains__(self, cls) -> bool:
        return bool(self.get(cls))
