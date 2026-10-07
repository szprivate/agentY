"""canvas_edit — structural edits to the graph open in the user's canvas.

``set_canvas_node_params`` writes values and ``delete_canvas_nodes`` removes
nodes; neither can add a node or change a wire, so "add a hires-fix pass",
"preview instead of saving" or "upscale before saving" could only be answered
by telling the user to do it by hand. This module is the missing third edit:
a list of operations — add a node, connect an output to an input, cut an
input's wire — checked against ComfyUI's ``/object_info`` and applied to the
API-format canvas graph the turn holds.

Pure on purpose: no I/O, no pipeline. The pipeline applies the result to its
copy of the canvas (so ``get_canvas_node`` reads what the edit made) and pushes
one ``{"op": "edit_graph"}`` canvas patch, which the panel replays on the live
graph — and which a headless client (the benchmark harness) can replay on its
own copy the same way.

Operations, applied in order, all or nothing::

    {"op": "add", "class_type": "LatentUpscaleBy", "ref": "up",
     "params": {"scale_by": 1.5}, "near": "5"}
    {"op": "connect", "from": "5", "output": 0, "to": "up", "input": "samples"}
    {"op": "disconnect", "node": "6", "input": "samples"}

``ref`` names a node added in the same call so later operations can wire it;
``output`` is a slot index or the output's name or type.
"""

from __future__ import annotations

import copy

_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def _specs(schema: dict) -> dict:
    inp = (schema or {}).get("input") or {}
    out = {}
    out.update(inp.get("optional") or {})
    out.update(inp.get("required") or {})
    return out


def _spec_type(spec):
    return spec[0] if isinstance(spec, (list, tuple)) and spec else None


def _is_dynamic(spec) -> bool:
    t = _spec_type(spec)
    return isinstance(t, str) and t.startswith("COMFY_") and t.endswith("_V3")


def _types(t) -> set:
    if isinstance(t, list):
        return {"COMBO"}
    return {p.strip() for p in str(t or "").split(",") if p.strip()}


def _compatible(out_t, in_t) -> bool:
    a, b = _types(out_t), _types(in_t)
    return "*" in a or "*" in b or bool(a & b) or ("COMBO" in b and a & {"COMBO", "STRING"})


def _combo_options(spec):
    t = _spec_type(spec)
    if isinstance(t, list):
        return t
    if t == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options") or []
    return None


def _default(spec):
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" in opts:
        return opts["default"]
    combo = _combo_options(spec)
    return combo[0] if combo else None


def _input_spec(specs: dict, name: str):
    """The spec for *name*, allowing V3 dynamic members ("values.a")."""
    if name in specs:
        return specs[name]
    head = name.split(".", 1)[0]
    if "." in name and head in specs and _is_dynamic(specs[head]):
        return ["*", {}]
    return None


def _member_spec(specs: dict, name: str, values: dict):
    """The spec of a dynamic combo's member — ``model.duration`` — for the option
    in force (``values[head]``, else the first), or None.

    `_input_spec` only knows such a name is allowed; without its own spec a
    value written there was never checked, and a Wan 3.0 ``model.duration`` of
    ``5`` went onto the canvas where the menu holds ``"5"`` — which ComfyUI
    refuses ("Value not in list") when the graph is queued."""
    head, _, rest = name.partition(".")
    spec = specs.get(head)
    if not rest or _spec_type(spec) != "COMFY_DYNAMICCOMBO_V3" or len(spec) < 2:
        return None
    options = (spec[1] or {}).get("options") or []
    key = (values or {}).get(head)
    opt = next((o for o in options if isinstance(o, dict) and o.get("key") == key), None)
    if opt is None and options and isinstance(options[0], dict):
        opt = options[0]
    sub = (opt or {}).get("inputs") or {}
    return {**(sub.get("optional") or {}), **(sub.get("required") or {})}.get(rest)


def _dynamic_defaults(name: str, spec, key=None) -> dict:
    """A required dynamic combo's value and its option's member defaults —
    ``{"model": "wan3.0-video", "model.resolution": "1080P", …}``.

    Left out, an added Wan 3.0 node had no `model` at all, so nothing on it
    showed a prompt, a resolution or a duration: the agent read the node as
    having none ("animates purely from the first frame") and stored that as a
    fact. And ComfyUI refuses the graph without the value."""
    options = [o for o in ((spec[1] or {}).get("options") or []) if isinstance(o, dict)] \
        if len(spec) > 1 and isinstance(spec[1], dict) else []
    opt = next((o for o in options if o.get("key") == key), None) or (options[0] if options else None)
    if opt is None:
        return {}
    out = {name: opt.get("key")}
    for member, mspec in ((opt.get("inputs") or {}).get("required") or {}).items():
        mt = _spec_type(mspec)
        if isinstance(mt, list) or mt in _WIDGET_TYPES:
            d = _default(mspec)
            if d is not None:
                out[f"{name}.{member}"] = d
    return out


def coerce_value(spec, value):
    """``(value, error)``: *value* as the input takes it, or the reason it can't.

    The usual slip is the type, not the value: a number written into a menu of
    strings ("5" for 5) or the other way round, which ComfyUI rejects although
    the option is there. Those are matched by their text. Number inputs take a
    numeric string as the number."""
    combo = _combo_options(spec)
    if combo is not None:
        if value in combo:
            return value, None
        same = next((o for o in combo if str(o) == str(value)), None)
        if same is not None:
            return same, None
        near = [o for o in combo if str(value).lower() in str(o).lower()][:5]
        return value, (f"= {value!r} is not an option"
                       + (f"; close: {near}" if near else f"; e.g. {combo[:5]}"))
    t = _spec_type(spec)
    if isinstance(value, str) and not isinstance(value, bool) and t in ("INT", "FLOAT"):
        try:
            num = float(value.strip())
            value = int(num) if t == "INT" and num.is_integer() else num
        except ValueError:
            return value, f"= {value!r} is not a number"
    # Out of range is refused at queue time too: ImageBatchMulti's inputcount
    # set to 1 (its minimum is 2) would have stopped every branch it feeds.
    if t in ("INT", "FLOAT") and isinstance(value, (int, float)) and not isinstance(value, bool):
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        lo, hi = opts.get("min"), opts.get("max")
        if lo is not None and value < lo:
            return value, f"= {value!r} is below its minimum of {lo}"
        if hi is not None and value > hi:
            return value, f"= {value!r} is above its maximum of {hi}"
    return value, None


def coerce_params(schema: dict, params: dict, current: dict | None = None) -> tuple[dict, list[str]]:
    """``(params, errors)``: *params* for a node of *schema*, each value as its
    input takes it. Names the schema does not know pass through untouched
    (the frontend has widgets of its own, e.g. ``control_after_generate``)."""
    specs = _specs(schema)
    values = {**(current or {}), **params}
    out, errors = {}, []
    for name, value in params.items():
        spec = specs.get(name) if name in specs else _member_spec(specs, name, values)
        if spec is None or _is_link(value):
            out[name] = value
            continue
        fixed, err = coerce_value(spec, value)
        if err:
            errors.append(f"{name} {err}")
        out[name] = fixed
    return out, errors


def _next_id(graph: dict) -> int:
    nums = [int(k) for k in graph if str(k).isdigit()]
    return (max(nums) if nums else 0) + 1


def unwired_required(graph: dict, node_id: str, object_info: dict) -> list[str]:
    """Required LINK inputs of *node_id* that nothing feeds — why it won't run."""
    node = graph.get(str(node_id)) or {}
    schema = object_info.get(node.get("class_type")) or {}
    req = ((schema.get("input") or {}).get("required") or {})
    inputs = node.get("inputs") or {}
    out = []
    for name, spec in req.items():
        t = _spec_type(spec)
        is_value = isinstance(t, list) or t in _WIDGET_TYPES or _is_dynamic(spec)
        if not is_value and name not in inputs:
            out.append(f"{name} ({t})")
    return out


def plan(graph: dict, ops: list, object_info: dict) -> dict:
    """Check and apply *ops* to a copy of *graph*.

    Returns ``{"ok": bool, "errors": [...], "graph": new_graph, "ops": [...],
    "added": {ref: node_id}, "touched": [node ids]}``. On any error nothing is
    applied (``graph`` is the input, unchanged) — a half-made edit is worse than
    none, because the next call cannot tell which half happened.
    """
    g = copy.deepcopy(graph if isinstance(graph, dict) else {})
    errors: list[str] = []
    resolved: list[dict] = []
    added: dict[str, str] = {}
    touched: list[str] = []

    def node_of(ref) -> str | None:
        r = str(ref).strip().lstrip("#")
        if r in added:
            return added[r]
        return r if r in g else None

    if not isinstance(ops, list) or not ops:
        return {"ok": False, "errors": ["ops must be a non-empty list of operations."],
                "graph": graph, "ops": [], "added": {}, "touched": []}

    for i, op in enumerate(ops, 1):
        where = f"op {i}"
        if not isinstance(op, dict):
            errors.append(f"{where}: each operation is an object, got {type(op).__name__}.")
            continue
        kind = str(op.get("op") or "").lower()

        if kind == "add":
            cls = str(op.get("class_type") or "")
            schema = object_info.get(cls)
            if not schema:
                errors.append(f"{where}: node type '{cls}' is not installed in this ComfyUI.")
                continue
            specs = _specs(schema)
            params = op.get("params") or {}
            if not isinstance(params, dict):
                errors.append(f"{where}: params must be an object of widget -> value.")
                continue
            inputs = {}
            for name, spec in ((schema.get("input") or {}).get("required") or {}).items():
                t = _spec_type(spec)
                if isinstance(t, list) or t in _WIDGET_TYPES:
                    d = _default(spec)
                    if d is not None:
                        inputs[name] = d
                elif t == "COMFY_DYNAMICCOMBO_V3":
                    inputs.update(_dynamic_defaults(name, spec, params.get(name)))
            bad = False
            for name in params:
                if _input_spec(specs, name) is None:
                    errors.append(f"{where}: {cls} has no input '{name}' "
                                  f"(it has: {', '.join(sorted(specs)) or 'none'}).")
                    bad = True
            params, wrong = coerce_params(schema, params, inputs)
            for w in wrong:
                errors.append(f"{where}: {cls}.{w}.")
                bad = True
            if bad:
                continue
            inputs.update(params)
            defaults = {k: v for k, v in inputs.items() if k not in params}
            nid = str(_next_id(g))
            g[nid] = {"class_type": cls, "inputs": inputs}
            ref = str(op.get("ref") or nid)
            added[ref] = nid
            added[nid] = nid
            near = node_of(op.get("near")) if op.get("near") is not None else None
            resolved.append({"op": "add", "node_id": nid, "class_type": cls,
                             "params": params, "defaults": defaults, "near": near})
            touched.append(nid)

        elif kind == "connect":
            src, dst = node_of(op.get("from")), node_of(op.get("to"))
            name = str(op.get("input") or "")
            if src is None or dst is None:
                missing = [str(op.get(k)) for k, v in (("from", src), ("to", dst)) if v is None]
                errors.append(f"{where}: no node {', '.join(missing)} on the canvas (or added earlier in this call).")
                continue
            s_schema = object_info.get(g[src].get("class_type")) or {}
            d_schema = object_info.get(g[dst].get("class_type")) or {}
            outs = list(s_schema.get("output") or [])
            names = list(s_schema.get("output_name") or [])
            want = op.get("output", 0)
            slot = None
            if isinstance(want, int) or (isinstance(want, str) and want.isdigit()):
                slot = int(want)
            else:
                for j, (t, n) in enumerate(zip(outs, names or outs)):
                    if str(want) in (str(t), str(n)):
                        slot = j
                        break
            if slot is None or not (0 <= slot < len(outs)):
                listing = ", ".join(f"{j}={n}({t})" for j, (t, n) in enumerate(zip(outs, names or outs)))
                errors.append(f"{where}: #{src} {g[src]['class_type']} has no output {want!r} "
                              f"(outputs: {listing or 'none'}).")
                continue
            spec = _input_spec(_specs(d_schema), name)
            if spec is None:
                errors.append(f"{where}: #{dst} {g[dst]['class_type']} has no input '{name}' "
                              f"(it has: {', '.join(sorted(_specs(d_schema))) or 'none'}).")
                continue
            if not _compatible(outs[slot], _spec_type(spec)):
                errors.append(f"{where}: #{src} output {slot} is {outs[slot]}, but "
                              f"#{dst}.{name} takes {_spec_type(spec)}.")
                continue
            g[dst].setdefault("inputs", {})[name] = [src, slot]
            resolved.append({"op": "connect", "from": src, "output": slot, "to": dst, "input": name})
            touched += [src, dst]

        elif kind == "disconnect":
            nid = node_of(op.get("node"))
            name = str(op.get("input") or "")
            if nid is None:
                errors.append(f"{where}: no node {op.get('node')} on the canvas.")
                continue
            if not _is_link((g[nid].get("inputs") or {}).get(name)):
                errors.append(f"{where}: #{nid}.{name} is not wired, so there is nothing to disconnect.")
                continue
            del g[nid]["inputs"][name]
            resolved.append({"op": "disconnect", "node": nid, "input": name})
            touched.append(nid)

        else:
            errors.append(f"{where}: unknown op {op.get('op')!r} — use add, connect or disconnect "
                          f"(delete_canvas_nodes removes nodes, set_canvas_node_params sets values).")

    if errors:
        return {"ok": False, "errors": errors, "graph": graph, "ops": [], "added": {}, "touched": []}
    refs = {k: v for k, v in added.items() if k != v}
    return {"ok": True, "errors": [], "graph": g, "ops": resolved, "added": refs,
            "touched": list(dict.fromkeys(touched))}


def ops_for_workflow(workflow: dict, prefix: str = "wf") -> list:
    """A built workflow (API format) as the ops that put it on a canvas, wired.

    One ``add`` per node - its values as ``params`` - and one ``connect`` per
    link, in an order :func:`plan` can apply: every node before the wires, and
    each node after the one it is placed beside. Nothing is decided here that
    the workflow does not already say: the model is not asked to restate a
    workflow it has just built, node by node and wire by wire.

    Raises ``ValueError`` for anything that is not an API-format workflow.
    """
    if not isinstance(workflow, dict) or isinstance(workflow.get("nodes"), list):
        raise ValueError("not an API-format workflow (a saved canvas graph cannot be inserted this way)")
    nodes = {str(k): v for k, v in workflow.items()
             if isinstance(v, dict) and v.get("class_type")}
    if not nodes:
        raise ValueError("the workflow has no nodes")

    def sources(nid: str) -> list:
        return [str(v[0]) for v in (nodes[nid].get("inputs") or {}).values()
                if _is_link(v) and str(v[0]) in nodes]

    # Upstream first, so a node can be placed beside the one that feeds it.
    order, seen = [], set()

    def visit(nid: str, trail: tuple = ()) -> None:
        if nid in seen or nid in trail:
            return
        for src in sources(nid):
            visit(src, trail + (nid,))
        seen.add(nid)
        order.append(nid)

    def key(n: str):
        return (0, int(n)) if n.isdigit() else (1, n)

    for nid in sorted(nodes, key=key):
        visit(nid)

    ref = {nid: f"{prefix}{nid}" for nid in nodes}
    ops: list = []
    for nid in order:
        node = nodes[nid]
        params = {k: v for k, v in (node.get("inputs") or {}).items() if not _is_link(v)}
        op = {"op": "add", "class_type": node["class_type"], "ref": ref[nid], "params": params}
        feeds = sources(nid)
        if feeds:
            op["near"] = ref[feeds[0]]
        ops.append(op)
    for nid in order:
        for name, value in (nodes[nid].get("inputs") or {}).items():
            if _is_link(value) and str(value[0]) in nodes:
                ops.append({"op": "connect", "from": ref[str(value[0])], "output": int(value[1]),
                            "to": ref[nid], "input": name})
    return ops


def apply_patch(graph: dict, patch: dict) -> dict:
    """Replay a pushed canvas patch on an API-format graph (in place; returned).

    What the panel does to the live graph, for a client that only has the API
    form — the benchmark harness, and the pipeline's own copy of the canvas.
    """
    op = patch.get("op")
    if op is None and isinstance(patch.get("params"), dict):
        node = graph.get(str(patch.get("node_id")))
        if node is not None:
            ins = node.setdefault("inputs", {})
            for k, v in patch["params"].items():
                if not _is_link(ins.get(k)):
                    ins[k] = v
    elif op == "delete_nodes":
        gone = {str(n) for n in patch.get("node_ids") or []}
        for n in gone:
            graph.pop(n, None)
        for node in graph.values():
            ins = node.get("inputs") or {}
            for k in [k for k, v in ins.items() if _is_link(v) and str(v[0]) in gone]:
                del ins[k]
    elif op == "edit_graph":
        for o in patch.get("ops") or []:
            if o.get("op") == "add":
                graph[str(o["node_id"])] = {"class_type": o["class_type"],
                                            "inputs": dict(o.get("defaults") or {}, **(o.get("params") or {}))}
            elif o.get("op") == "connect" and str(o.get("to")) in graph:
                graph[str(o["to"])].setdefault("inputs", {})[o["input"]] = [str(o["from"]), int(o["output"])]
            elif o.get("op") == "disconnect" and str(o.get("node")) in graph:
                (graph[str(o["node"])].get("inputs") or {}).pop(o.get("input"), None)
    return graph
